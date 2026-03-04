import logging
import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CONTENTFUL_ACCESS_TOKEN = os.getenv("CONTENTFUL_ACCESS_TOKEN")
CONTENTFUL_SPACE_ID = os.getenv("CONTENTFUL_SPACE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

assert PINECONE_API_KEY, "PINECONE_API_KEY not set in .env"
assert CONTENTFUL_ACCESS_TOKEN, "CONTENTFUL_ACCESS_TOKEN not set in .env"
assert CONTENTFUL_SPACE_ID, "CONTENTFUL_SPACE_ID not set in .env"
assert HF_TOKEN, "HF_TOKEN not set in .env"

# Pinecone settings
PINECONE_INDEX_NAME = "platform"
PINECONE_NAMESPACE = "search"

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIMENSION = 1024
