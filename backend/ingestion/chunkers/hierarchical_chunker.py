"""
Strategy 6 — Hierarchical chunker  ★ for policy / legal documents ★

Produces three tiers of chunks from the same document:
  Parent     : 2048 tokens  — broad context window
  Child      : 512  tokens  — standard retrieval unit
  Grandchild : 128  tokens  — fine-grained citation window

Every child carries parent_chunk_id and every grandchild carries both
parent_chunk_id and grandparent_chunk_id so the retrieval layer can
expand context up or down the hierarchy at query time.

Chunk ID convention:
  parent     : {doc_id}:P{parent_idx}
  child      : {doc_id}:C{child_idx}
  grandchild : {doc_id}:G{grandchild_idx}
"""
from __future__ import annotations

import logging

from backend.ingestion.chunkers.base_chunker import BaseChunker
from backend.models.chunk import Chunk, ChunkingConfig, ChunkingStrategy, ChunkType
from backend.models.parsed_document import ParsedDocument, ParsedPage
from backend.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class HierarchicalChunker(BaseChunker):
    """
    Three-tier hierarchical chunker for policy and legal documents.

    The public `chunk()` method is overridden here because the output
    contains multiple ChunkType values (PARENT / CHILD / GRANDCHILD)
    that cannot be modelled by the single-type base flow.
    """

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.HIERARCHICAL

    def chunk_text(
        self,
        text:   str,
        config: ChunkingConfig,
        doc:    ParsedDocument,
        pages:  list[ParsedPage],
    ) -> list[str]:
        # chunk_text is only used by the base class — we override chunk()
        # so this method is only called as a fallback. Return parent splits.
        return self._split_by_tokens(text, config.parent_chunk_size, overlap=0)

    # ── Override chunk() to produce three-tier output ─────────────────────────

    def chunk(
        self,
        doc:    ParsedDocument,
        config: ChunkingConfig,
        pages:  list[ParsedPage] | None = None,
    ) -> list[Chunk]:
        if pages is None:
            pages = doc.pages

        text = self._pages_to_text(pages)
        if not text.strip():
            return []

        title_prefix = self._build_title_prefix(doc, config)
        headings     = self._extract_headings(text)
        page_numbers = [p.page_number for p in pages]

        # ── Tier 1: Parents ────────────────────────────────────────────────────
        parent_texts = _recursive_split(
            text, ["\n\n", "\n", " "], config.parent_chunk_size, overlap=0
        )

        all_chunks: list[Chunk] = []
        global_child_idx       = 0
        global_grandchild_idx  = 0

        for p_idx, p_text in enumerate(parent_texts):
            if not p_text.strip():
                continue

            p_id  = f"{doc.doc_id}:P{p_idx}"
            p_pos = text.find(p_text[:60].strip())
            p_heading = self._find_nearest_heading(max(0, p_pos), headings)
            p_prefix  = (
                f"{title_prefix} | {p_heading}" if p_heading and title_prefix
                else p_heading or title_prefix
            )

            parent_chunk = self._make_chunk(
                text=p_text,
                doc=doc,
                config=config,
                chunk_index=p_idx,
                page_numbers=page_numbers,
                title_prefix=p_prefix,
                chunk_type=ChunkType.PARENT,
                extras={"tier": "parent", "parent_index": p_idx},
            )
            # Override the auto-generated chunk_id with our naming convention
            object.__setattr__(parent_chunk, "chunk_id", p_id)
            all_chunks.append(parent_chunk)

            # ── Tier 2: Children (split each parent) ──────────────────────────
            child_texts = _recursive_split(
                p_text, ["\n\n", "\n", " "], config.child_chunk_size, overlap=0
            )

            for c_text in child_texts:
                if not c_text.strip():
                    continue

                c_id = f"{doc.doc_id}:C{global_child_idx}"
                c_pos = text.find(c_text[:60].strip(), max(0, p_pos - 20))
                c_heading = self._find_nearest_heading(max(0, c_pos), headings)
                c_prefix  = (
                    f"{p_prefix} | {c_heading}" if c_heading and p_prefix
                    else c_heading or p_prefix
                )

                child_chunk = self._make_chunk(
                    text=c_text,
                    doc=doc,
                    config=config,
                    chunk_index=global_child_idx,
                    page_numbers=page_numbers,
                    title_prefix=c_prefix,
                    chunk_type=ChunkType.CHILD,
                    parent_chunk_id=p_id,
                    extras={
                        "tier": "child",
                        "parent_id": p_id,
                        "child_index": global_child_idx,
                    },
                )
                object.__setattr__(child_chunk, "chunk_id", c_id)
                all_chunks.append(child_chunk)

                # ── Tier 3: Grandchildren (split each child) ──────────────────
                gc_texts = _recursive_split(
                    c_text,
                    ["\n", " "],
                    config.grandchild_chunk_size,
                    overlap=0,
                )

                for gc_text in gc_texts:
                    if not gc_text.strip():
                        continue
                    gc_id = f"{doc.doc_id}:G{global_grandchild_idx}"

                    gc_chunk = self._make_chunk(
                        text=gc_text,
                        doc=doc,
                        config=config,
                        chunk_index=global_grandchild_idx,
                        page_numbers=page_numbers,
                        title_prefix=c_prefix,
                        chunk_type=ChunkType.GRANDCHILD,
                        parent_chunk_id=c_id,
                        grandparent_chunk_id=p_id,
                        extras={
                            "tier": "grandchild",
                            "parent_id": c_id,
                            "grandparent_id": p_id,
                            "grandchild_index": global_grandchild_idx,
                        },
                    )
                    object.__setattr__(gc_chunk, "chunk_id", gc_id)
                    all_chunks.append(gc_chunk)
                    global_grandchild_idx += 1

                global_child_idx += 1

        parents     = [c for c in all_chunks if c.chunk_type == ChunkType.PARENT]
        children    = [c for c in all_chunks if c.chunk_type == ChunkType.CHILD]
        grandchildren = [c for c in all_chunks if c.chunk_type == ChunkType.GRANDCHILD]

        logger.info(
            "[Hierarchical] '%s' → %d parents, %d children, %d grandchildren",
            doc.filename, len(parents), len(children), len(grandchildren),
        )
        return all_chunks


# ─── Shared recursive split (keeps hierarchical_chunker self-contained) ───────

def _recursive_split(
    text:       str,
    separators: list[str],
    chunk_size: int,
    overlap:    int,
) -> list[str]:
    if count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        chars = chunk_size * 4
        return [text[i:i + chars] for i in range(0, len(text), chars) if text[i:i + chars].strip()]

    sep    = separators[0]
    rest   = separators[1:]
    splits = text.split(sep) if sep else list(text)

    chunks:  list[str] = []
    current: list[str] = []
    cur_tok             = 0

    for piece in splits:
        if not piece:
            continue
        pt = count_tokens(piece)
        if pt > chunk_size:
            if current:
                chunks.append(sep.join(current))
                current, cur_tok = [], 0
            chunks.extend(_recursive_split(piece, rest, chunk_size, overlap))
            continue
        if cur_tok + pt > chunk_size and current:
            chunks.append(sep.join(current))
            current, cur_tok = [], 0
        current.append(piece)
        cur_tok += pt

    if current:
        chunks.append(sep.join(current))

    return [c for c in chunks if c.strip()]
