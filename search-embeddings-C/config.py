import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CONTENTFUL_ACCESS_TOKEN = os.getenv("CONTENTFUL_ACCESS_TOKEN")
CONTENTFUL_SPACE_ID = os.getenv("CONTENTFUL_SPACE_ID")
GETSTREAM_API_KEY = os.getenv("GETSTREAM_API_KEY")
GETSTREAM_API_SECRET = os.getenv("GETSTREAM_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert PINECONE_API_KEY, "PINECONE_API_KEY not set in .env"
assert CONTENTFUL_ACCESS_TOKEN, "CONTENTFUL_ACCESS_TOKEN not set in .env"
assert CONTENTFUL_SPACE_ID, "CONTENTFUL_SPACE_ID not set in .env"
assert GETSTREAM_API_KEY, "GETSTREAM_API_KEY not set in .env"
assert GETSTREAM_API_SECRET, "GETSTREAM_API_SECRET not set in .env"
assert OPENAI_API_KEY, "OPENAI_API_KEY not set in .env"

# Pinecone settings
PINECONE_INDEX_NAME = "platform-v3"
PINECONE_NAMESPACE = "search"

# Dense embedding model (Pinecone-hosted multilingual-e5-large)
DENSE_MODEL = "multilingual-e5-large"
DENSE_DIM = 1024
VECTOR_TYPE = "dense"
METRIC = "cosine"
MODALITY = "text"

# Supported locales for embedding
SUPPORTED_LOCALES = [
    "en",
    "en-us",
    "en-gb",
    "fr",
    "fr-fr",
    "es",
    "es-es",
    "de",
    "de-de",
    "it",
    "it-it",
    "pt",
    "pt-pt",
    "nl",
    "nl-nl",
    "pl",
    "pl-pl",
    "sv",
    "sv-se",
    "da",
    "da-dk",
    "fi",
    "fi-fi",
    "no",
    "no-no",
    "cs",
    "cs-cz",
    "el",
    "el-gr",
    "hu",
    "hu-hu",
    "ro",
    "ro-ro",
    "sk",
    "sk-sk",
]
