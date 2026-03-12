import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import OPENAI_API_KEY

log = logging.getLogger(__name__)

MODEL = "gpt-4o"
# ~100k chars fits comfortably within gpt-4o's 128k token window
STUFF_CHAR_LIMIT = 100_000
CHUNK_SIZE = 80_000
CHUNK_OVERLAP = 2_000

_llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0.65)

_SYSTEM = (
    "You are an expert analyst helping build a knowledge base for a RAG system used by public servants. "
    "Your summaries will be converted into embeddings and retrieved when public servants ask questions "
    "about communities on a government collaboration platform. "
    "Write in clear, professional prose. Be descriptive and specific — vague summaries reduce retrieval quality."
)

_SUMMARISE_PROMPT = (
    "Community title: {title}\n\n"
    "Analyse the community title and the discussion below, then write a descriptive summary that:\n"
    "- Explains what this community is about and who it serves, using the title as the primary signal "
    "if the discussion content is sparse or absent\n"
    "- Highlights the key topics, themes, and subject areas discussed\n"
    "- Captures the main concerns, challenges, and questions raised by members\n"
    "- Notes any recurring debates, shared goals, or areas of consensus\n"
    "- Uses concrete, domain-specific language that would match how public servants phrase questions\n\n"
    "Community discussion:\n{text}"
)

_COMBINE_PROMPT = (
    "You have been given partial summaries of sections from a single community discussion. "
    "Merge them into one coherent, descriptive summary that:\n"
    "- Describes what this community is about and who it serves\n"
    "- Covers all key topics, themes, and subject areas across the sections\n"
    "- Captures the main concerns, challenges, and questions raised by members\n"
    "- Notes recurring debates, shared goals, or areas of consensus\n"
    "- Uses concrete, domain-specific language that would match how public servants phrase questions\n\n"
    "Partial summaries:\n{text}"
)


def _call(prompt: str) -> str:
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
    return _llm.invoke(messages).content.strip()


def summarise_text(text: str, title: str = "") -> str:
    """Summarise community feed text using gpt-4o.

    Returns an empty string if the text is shorter than 10 characters.
    Automatically switches to map_reduce chunking for texts that exceed
    the context window limit.
    """
    if len(text) < 10:
        return ""

    if len(text) <= STUFF_CHAR_LIMIT:
        return _call(_SUMMARISE_PROMPT.format(title=title, text=text))

    log.info("Text length %d exceeds limit, using map_reduce summarisation", len(text))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    partial_summaries = [
        _call(_SUMMARISE_PROMPT.format(title=title, text=chunk)) for chunk in chunks
    ]
    combined = "\n\n".join(partial_summaries)
    return _call(_COMBINE_PROMPT.format(text=combined))
