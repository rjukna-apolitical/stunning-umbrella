"""
Fit BM25 on the full multilingual corpus and save stats.

Run this once before embedding (after setup_index.py):
    python setup_bm25.py

The fitted stats are saved to data/bm25_corpus_stats.json and loaded
automatically by the embed scripts.
"""
import logging

from config import BM25_STATS_PATH
from modules.bm25 import SUPPORTED_LOCALES, MultilingualBM25
from modules.contentful import get_all_entries
from modules.logger import setup_logging
from modules.richtext import extract_values

log = logging.getLogger(__name__)

# Content types to include in BM25 corpus fitting.
# More content = more accurate IDF statistics.
CORPUS_CONTENT_TYPES = {
    "solutionArticle": "body",
    "event": "description",
}


def collect_corpus() -> list[dict]:
    """Fetch all content across all locales and return as BM25 training docs."""
    corpus: list[dict] = []

    for content_type, text_field in CORPUS_CONTENT_TYPES.items():
        log.info("Collecting %s entries for BM25 corpus...", content_type)
        entry_count = 0

        for entry in get_all_entries(content_type):
            fields = entry.raw.get("fields", {})
            title_by_locale = fields.get("title", {})
            body_by_locale = fields.get(text_field, {})

            for locale in SUPPORTED_LOCALES:
                title = title_by_locale.get(locale, "")
                body = body_by_locale.get(locale, "")
                if not isinstance(body, str):
                    body = ""
                text = f"{title}. {body}".strip()
                if text and text != ".":
                    corpus.append({"text": text, "locale": locale})

            entry_count += 1

        log.info("Collected %d %s entries", entry_count, content_type)

    # Also collect course journey overview text (per locale)
    log.info("Collecting course journey entries for BM25 corpus...")
    journey_count = 0
    seen_journeys: set[str] = set()

    for course_entry in get_all_entries("course", include=0):
        fields = course_entry.raw.get("fields", {})
        journeys = fields.get("journeys", {}).get("en", [])

        for journey_ref in journeys:
            journey_id = journey_ref["sys"]["id"]
            if journey_id in seen_journeys:
                continue
            seen_journeys.add(journey_id)

            try:
                from modules.contentful import getEntry
                journey = getEntry(journey_id)
                jfields = journey.raw.get("fields", {})

                left_by_locale = jfields.get("overviewLeftColumn", {})
                right_by_locale = jfields.get("overviewRightColumn", {})
                title_by_locale = jfields.get("title", {})

                for locale in SUPPORTED_LOCALES:
                    title = title_by_locale.get(locale, "")
                    left = extract_values(left_by_locale.get(locale, {}).get("content", []))
                    right = extract_values(right_by_locale.get(locale, {}).get("content", []))
                    text = f"{title}. " + " ".join(left + right)
                    if text.strip() and text.strip() != ".":
                        corpus.append({"text": text, "locale": locale})

                journey_count += 1
            except Exception as e:
                log.warning("Failed to fetch journey %s — skipping (%s: %s)", journey_id, type(e).__name__, e)

    log.info("Collected %d unique course journeys", journey_count)

    return corpus


def main():
    setup_logging(log_file="fit-bm25.log")
    log.info("Starting BM25 corpus fitting...")

    corpus = collect_corpus()
    log.info("Total corpus: %d documents across all locales", len(corpus))

    bm25 = MultilingualBM25()
    bm25.fit(corpus)
    bm25.save_stats(BM25_STATS_PATH)

    log.info("BM25 fitting complete. Stats saved to %s", BM25_STATS_PATH)


if __name__ == "__main__":
    main()
