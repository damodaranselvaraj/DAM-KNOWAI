"""
Data formatting helpers used across all frontend pages and components.
"""
from __future__ import annotations
from datetime import datetime, timezone
from frontend.utils.constants import COLOR_SCHEME, SCORE_THRESHOLDS


def format_timestamp(iso_str: str | None) -> str:
    """Convert an ISO-8601 string to 'Jan 15, 2024 at 10:42 AM'."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y at %I:%M %p")
    except (ValueError, AttributeError):
        return iso_str


def format_timestamp_short(iso_str: str | None) -> str:
    """Convert an ISO-8601 string to 'HH:MM AM/PM'."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%I:%M %p")
    except (ValueError, AttributeError):
        return ""


def format_file_size(size_bytes: int | float) -> str:
    """Convert byte count to human-readable string: '1.2 MB'."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.1f} GB"


def format_number(value: int | float) -> str:
    """Format a large integer with thousands separators: '12,430'."""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


def format_score(score: float) -> str:
    """Return a percentage string with 1 decimal place: '94.3%'."""
    try:
        return f"{score * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def get_score_color(score: float) -> str:
    """Return a hex colour based on the similarity score value."""
    if score >= SCORE_THRESHOLDS["excellent"]:
        return COLOR_SCHEME["success"]   # green
    elif score >= SCORE_THRESHOLDS["good"]:
        return COLOR_SCHEME["warning"]   # yellow
    elif score >= SCORE_THRESHOLDS["fair"]:
        return "#FF9800"                 # orange
    return COLOR_SCHEME["danger"]        # red


def get_score_badge(score: float) -> str:
    """Return a coloured emoji badge for the score tier."""
    if score >= SCORE_THRESHOLDS["excellent"]:
        return "🟢"
    elif score >= SCORE_THRESHOLDS["good"]:
        return "🟡"
    elif score >= SCORE_THRESHOLDS["fair"]:
        return "🟠"
    return "🔴"


def truncate_text(text: str, max_len: int = 120) -> str:
    """Truncate text with an ellipsis if it exceeds max_len characters."""
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len].rstrip() + "…"


def parse_streaming_chunk(chunk: str) -> str:
    """Extract text payload from a raw SSE 'data: ...' line."""
    if chunk.startswith("data:"):
        payload = chunk[5:].strip()
        if payload == "[DONE]":
            return ""
        return payload + " "
    return ""


def build_citation_markdown(citation: dict, index: int = 1) -> str:
    """Build a compact markdown string for a single citation."""
    score_badge = get_score_badge(citation.get("score", 0))
    score_pct = format_score(citation.get("score", 0))
    snippet = truncate_text(citation.get("snippet", ""), 100)
    return (
        f"**[{index}] {citation.get('doc_name', 'Unknown')}**  \n"
        f"📄 Page {citation.get('page_number', '?')} &nbsp;|&nbsp; "
        f"🗓 {citation.get('published_at', '—')} &nbsp;|&nbsp; "
        f"{score_badge} Score: {score_pct} &nbsp;|&nbsp; "
        f"Chunk #{citation.get('chunk_index', '?')}  \n"
        f"*{snippet}*"
    )


def estimate_chunks(token_count: int, chunk_size: int, overlap: int) -> int:
    """Estimate the number of chunks for a document of `token_count` tokens."""
    if chunk_size <= overlap:
        return token_count
    effective = chunk_size - overlap
    if effective <= 0:
        return 1
    return max(1, round(token_count / effective))


def get_file_icon(file_type: str) -> str:
    """Return an emoji icon for a given file extension."""
    icons = {
        "pdf":  "📄",
        "docx": "📝",
        "doc":  "📝",
        "csv":  "📊",
        "xlsx": "📊",
        "xls":  "📊",
        "html": "🌐",
        "htm":  "🌐",
        "txt":  "📃",
    }
    return icons.get(file_type.lower().lstrip("."), "📁")
