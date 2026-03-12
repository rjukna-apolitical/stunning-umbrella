"""
Docker entrypoint for search-embedding-c.

Usage:
    docker run ... search-embedding-c setup           # Create index
    docker run ... search-embedding-c clear <type>   # Clear embeddings
    docker run ... search-embedding-c embed <type>   # Create embeddings
"""

import sys

from modules.logger import setup_logging
from modules.pinecone_utils import delete_by_type


def main():
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python entrypoint.py <command>")
        print("Commands: setup, clear <type>, embed <type>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "setup":
        from setup_index import create_index, clear_namespace

        create_index()
        clear_namespace()

    elif command == "clear":
        if len(sys.argv) < 3:
            print("Usage: python entrypoint.py clear <type>")
            sys.exit(1)
        content_type = sys.argv[2]
        delete_by_type(content_type)

    elif command == "embed":
        if len(sys.argv) < 3:
            print("Usage: python entrypoint.py embed <type>")
            sys.exit(1)
        content_type = sys.argv[2]

        if content_type == "article":
            from embed.article import embed_article

            embed_article()
        elif content_type == "event":
            from embed.event import embed_event

            embed_event()
        elif content_type == "course":
            from embed.course import embed_course

            embed_course()
        elif content_type == "community":
            from embed.community import embed_community

            embed_community()
        else:
            print(f"Unknown content type: {content_type}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
