"""
Shared helper for turning CRAG's retrieved Documents into displayable
citation metadata.

Used in two places:
- nodes.py attaches this to each generated AIMessage's additional_kwargs,
  so citation data is persisted per-message (not just for the latest turn).
- chat_service.py falls back to it for any older, already-checkpointed
  messages that don't have this baked in yet.
"""

import os


def build_sources(docs):
    """Builds [{'index': 1, 'label': 'book.pdf (page 12)', 'snippet': ...}, ...]
    from a list of LangChain Documents, in the same order they were numbered
    in the prompt (so indexes line up with the [1], [2] markers in the text).
    """
    sources = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        raw_source = meta.get("source", "Unknown source")
        page = meta.get("page")

        # Local PDFs/TXTs store a full file path — show just the filename.
        # Web search results store a URL — show it as-is.
        if isinstance(raw_source, str) and raw_source.startswith(("http://", "https://")):
            label = raw_source
        else:
            label = os.path.basename(str(raw_source))

        if isinstance(page, int):
            label += f" (page {page + 1})"

        sources.append({
            "index": i,
            "label": label,
            "snippet": doc.page_content,
        })

    return sources