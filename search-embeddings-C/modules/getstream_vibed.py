import logging
import time
from typing import Generator

from getstream import Stream
from getstream.feeds.feeds import Feed

from config import GETSTREAM_API_KEY, GETSTREAM_API_SECRET

log = logging.getLogger(__name__)

client = Stream(api_key=GETSTREAM_API_KEY, api_secret=GETSTREAM_API_SECRET)

PAGE_SIZE = 25
MAX_RETRIES = 5
FEED_VIEW = "recent_with_pin"


def get_feed(feed_group: str, feed_id: str) -> Feed:
    """Return a Feed object for the given group and ID."""
    return client.feeds.feed(feed_group, feed_id)


def get_feed_activities(
    feed_group: str, feed_id: str, page_size: int = PAGE_SIZE
) -> Generator:
    """Paginate through all activities in a feed.

    Uses feed.get_or_create(view=FEED_VIEW) with cursor-based pagination.
    Yields one ActivityResponse at a time.
    """
    feed = get_feed(feed_group, feed_id)
    feed_ref = f"{feed_group}:{feed_id}"
    next_cursor = None
    total = 0

    while True:
        for attempt in range(MAX_RETRIES):
            try:
                response = feed.get_or_create(
                    view=FEED_VIEW,
                    limit=page_size,
                    next=next_cursor,
                )
                break
            except Exception as e:
                if "404" in str(e):
                    log.warning("Feed %s not found — skipping.", feed_ref)
                    return
                wait = 2**attempt
                log.warning(
                    "Error fetching activities for %s (attempt %d): %s. Retrying in %ds...",
                    feed_ref, attempt + 1, e, wait,
                )
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"Failed to fetch activities for {feed_ref} after {MAX_RETRIES} retries"
            )

        activities = response.data.activities or []
        for activity in activities:
            yield activity
            total += 1

        next_cursor = response.data.next
        if not next_cursor or not activities:
            break

        time.sleep(0.2)

    log.debug("Fetched %d activities for %s", total, feed_ref)


def get_activity_comments(
    activity_id: str, page_size: int = PAGE_SIZE
) -> Generator:
    """Paginate through all comments for an activity.

    Yields one ThreadedCommentResponse at a time.
    """
    next_cursor = None
    total = 0

    while True:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.feeds.get_comments(
                    object_id=activity_id,
                    object_type="activity",
                    limit=page_size,
                    next=next_cursor,
                )
                break
            except Exception as e:
                wait = 2**attempt
                log.warning(
                    "Error fetching comments for activity %s (attempt %d): %s. Retrying in %ds...",
                    activity_id, attempt + 1, e, wait,
                )
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"Failed to fetch comments for activity {activity_id} after {MAX_RETRIES} retries"
            )

        comments = response.data.comments or []
        for comment in comments:
            yield comment
            total += 1

        next_cursor = response.data.next
        if not next_cursor or not comments:
            break

        time.sleep(0.2)

    log.debug("Fetched %d comments for activity %s", total, activity_id)
