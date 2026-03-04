"""
Docker entrypoint — runs a single step of the multilingual embedding pipeline.

Usage:
    python entrypoint.py fit                     # fit BM25 corpus stats
    python entrypoint.py <content_type>          # embed
    python entrypoint.py clear <content_type>    # delete vectors by type

    content_type: article | event | course
"""
import sys

from modules.logger import setup_logging, get_logger

log = get_logger("entrypoint")

CONTENT_TYPES = {"article", "event", "course"}

EMBEDDERS = {
    "article": lambda: __import__("embed.article", fromlist=["embed_article"]).embed_article(),
    "event":   lambda: __import__("embed.event",   fromlist=["embed_event"]).embed_event(),
    "course":  lambda: __import__("embed.course",  fromlist=["embed_course"]).embed_course(),
}

# Map CLI content_type arg → Pinecone metadata content_type value
TYPE_MAP = {
    "article": "solutionArticle",
    "event":   "event",
    "course":  "course",
}


def main() -> None:
    args = sys.argv[1:]

    # fit
    if len(args) == 1 and args[0] == "fit":
        setup_logging(log_file="fit-bm25.log")
        log.info("Starting BM25 corpus fitting")
        try:
            from setup_bm25 import main as fit_main
            fit_main()
        except Exception:
            log.exception("BM25 fitting failed")
            sys.exit(1)
        return

    # clear <type>
    if len(args) == 2 and args[0] == "clear" and args[1] in CONTENT_TYPES:
        content_type = args[1]
        setup_logging()
        log.info("Clearing vectors for type: %s", content_type)
        try:
            from modules.pinecone_utils import delete_by_type
            delete_by_type(TYPE_MAP[content_type])
        except Exception:
            log.exception("Failed to clear vectors for: %s", content_type)
            sys.exit(1)
        return

    # embed <type>
    if len(args) == 1 and args[0] in EMBEDDERS:
        content_type = args[0]
        setup_logging(log_file=f"{content_type}-embedding.log")
        log.info("Starting embedding pipeline: %s", content_type)
        try:
            EMBEDDERS[content_type]()
            log.info("Embedding pipeline complete: %s", content_type)
        except Exception:
            log.exception("Embedding pipeline failed for: %s", content_type)
            sys.exit(1)
        return

    log.error("Usage:")
    log.error("  python entrypoint.py fit                  # fit BM25")
    log.error("  python entrypoint.py <type>               # embed")
    log.error("  python entrypoint.py clear <type>         # delete vectors")
    log.error("Available types: %s", " | ".join(CONTENT_TYPES))
    sys.exit(1)


if __name__ == "__main__":
    main()
