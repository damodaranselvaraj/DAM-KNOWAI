"""
Parsers sub-package.
Public exports for convenience imports elsewhere in the codebase.
"""
from backend.ingestion.parsers.base_parser       import BaseParser
from backend.ingestion.parsers.pymupdf_parser    import PyMuPDFParser
from backend.ingestion.parsers.docling_parser    import DoclingParser
from backend.ingestion.parsers.llamaindex_parser import LlamaIndexParser
from backend.ingestion.parsers.parser_router     import ParserRouter, router

__all__ = [
    "BaseParser",
    "PyMuPDFParser",
    "DoclingParser",
    "LlamaIndexParser",
    "ParserRouter",
    "router",
]
