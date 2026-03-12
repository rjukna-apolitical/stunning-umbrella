import pprint
from typing import Generator

import stream

from config import GETSTREAM_API_KEY, GETSTREAM_API_SECRET

pp = pprint.PrettyPrinter(indent=4)

client = stream.connect(GETSTREAM_API_KEY, GETSTREAM_API_SECRET, location="eu-west")

PAGE_SIZE = 25


def get_feed(feed_group: str, feed_id: str):
    """Return a Feed object for the given group and ID."""
    return client.feed(feed_group, feed_id)


def get_feed_activities(
    feed_group: str, feed_id: str, page_size: int = PAGE_SIZE
) -> Generator:
    """Paginate through all activities in a feed using cursor-based pagination.

    Yields one activity dict at a time.
    """
    feed = get_feed(feed_group, feed_id)
    id_lt = None

    while True:
        kwargs = {"limit": page_size, "reactions": {"recent": True, "counts": True}}
        if id_lt:
            kwargs["id_lt"] = id_lt

        response = feed.get(**kwargs)
        activities = response.get("results", [])

        for activity in activities:
            yield activity

        if len(activities) < page_size:
            break

        id_lt = activities[-1]["id"]


def extract_content(activity: dict) -> dict:
    """Extract text content and reaction messages from an activity."""
    original = activity.get("payload", {}).get("data", {}).get("original", {})

    # Collect comment reaction texts (reactions keyed by type, e.g. "comment", "like")
    reaction_messages = []
    for reaction_type, reactions in activity.get("latest_reactions", {}).items():
        for r in reactions:
            text = r.get("data", {}).get("text") or r.get("data", {}).get("body")
            if text:
                reaction_messages.append({"type": reaction_type, "text": text})

    return {
        "id": activity["id"],
        "foreign_id": activity.get("foreign_id"),
        "time": activity.get("time"),
        "title": original.get("title"),
        "body": original.get("body"),
        "locale": original.get("locale"),
        "reaction_messages": reaction_messages,
    }


def get_feed_text(feed_group: str, feed_id: str) -> str:
    """Return all activity content from a feed as a single combined text block."""
    parts = []
    for activity in get_feed_activities(feed_group, feed_id):
        content = extract_content(activity)
        chunk = "\n".join(filter(None, [content["title"], content["body"]]))
        if chunk:
            parts.append(chunk)
        for msg in content["reaction_messages"]:
            parts.append(msg["text"])
    return "\n\n".join(parts)


if __name__ == "__main__":
    text = get_feed_text("community", "ai-in-government")
    print(text)
