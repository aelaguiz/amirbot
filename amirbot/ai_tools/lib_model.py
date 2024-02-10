import logging
import os
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.globals import set_llm_cache
from langchain.indexes import SQLRecordManager, index
from langchain.cache import SQLiteCache
from langchain.vectorstores.pgvector import PGVector
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
import httpx

_vectordb = None
_embedding = None
_pinecone_index = None
_llm = None
_json_llm = None
_embedding = None
_db = None
_record_manager = None

OPENAI_API_KEY = OPENAI_MODEL = OPENAI_MAX_REQUESTS_PER_MINUTE = OPENAI_TEMPERATURE = None

# OPENAI_MAX_REQUESTS_PER_MINUTE=60
# OPENAI_MODEL="gpt-4-1106-preview"
# OPENAI_TEMPERATURE=0.25


# def init(model_name, api_key, db_connection_string, record_manager_connection_string, temp=0.5):
def init():
    global _llm
    global _embedding
    global _db
    global _record_manager
    global _json_llm
    global OPENAI_MODEL
    global OPENAI_MAX_REQUESTS_PER_MINUTE
    global OPENAI_TEMPERATURE

    OPENAI_MODEL = os.getenv("OPENAI_MODEL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MAX_REQUESTS_PER_MINUTE = os.getenv("OPENAI_MAX_REQUESTS_PER_MINUTE")
    OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")

    logger = logging.getLogger(__name__)

    if _llm:
        logger.warning("LLM already initialized, skipping")
        return _llm

    _llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)
    _embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, timeout=30)
    # _db = initialize_db(db_connection_string, record_manager_connection_string)
    # set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    _json_llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE, timeout=httpx.Timeout(15.0, read=60.0, write=10.0, connect=3.0), max_retries=0).bind(
        response_format= {
            "type": "json_object"
        }
    )


def get_embedding_fn():
    global _embedding

    logger = logging.getLogger(__name__)

    if not _embedding:
        logger.error("Embedding not initialized, call init() first")
        raise Exception("Embedding not initialized, call init() first")
    
    return _embedding

def initialize_db(db_connection_string, record_manager_connection_string, db_collection_name="docs"):
    global _db
    global _record_manager

    if _db:
        raise Exception("DB already initialized")

    _db = PGVector(
        embedding_function=get_embedding_fn(),
        collection_name=db_collection_name,
        connection_string=db_connection_string
    )

    namespace = f"pgvector/{db_collection_name}"
    _record_manager = SQLRecordManager(namespace, db_url=record_manager_connection_string)

    _record_manager.create_schema()

    return _db

def get_record_manager():
    global _record_manager

    logger = logging.getLogger(__name__)

    if not _record_manager:
        logger.error("Record manager not initialized, call initialize_db() first")
        raise Exception("Record manager not initialized, call initialize_db() first")

    return _record_manager
    
def get_vectordb():
    return _db

def get_llm():
    global _llm

    logger = logging.getLogger(__name__)

    if not _llm:
        logger.error("LLM not initialized, call init() first")
        raise Exception("LLM not initialized, call init() first")

    return _llm

def get_json_llm():
    global _json_llm

    logger = logging.getLogger(__name__)

    if not _json_llm:
        logger.error("JSON LLM not initialized, call init() first")
        raise Exception("JSON LLM not initialized, call init() first")

    return _json_llm
