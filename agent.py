import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import Tool
import streamlit as st
from agents.query_agent import QueryAgent
from agents.rewrite_nl_agent import RewriteNLAgent
from logger import logger, log_db_fallback


def get_secret(key: str, default: str = "") -> str:
    """Streamlit secrets veya ortam değişkenlerinden güvenli değer okur."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# --- 1. API ANAHTARLARI & ORTAM DEĞİŞKENLERİ ---
for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "OPENAI_MODEL_NAME"]:
    val = get_secret(key)
    if val:
        os.environ[key] = val

openai_model = get_secret("OPENAI_MODEL_NAME", "gpt-4o")

CLICKHOUSE_CORE_TABLES = ["tweet_predictions", "tweets", "users", "user_factors"]


def get_database_connection():
    """
    ClickHouse bağlantısını dener; ulaşılamazsa log_db_fallback çağırarak SQLite'a geçer.
    """
    ch_host = get_secret("CLICKHOUSE_HOST")
    
    if ch_host:
        ch_port = get_secret("CLICKHOUSE_PORT", "8123")
        ch_user = get_secret("CLICKHOUSE_USERNAME", "default")
        ch_pass = get_secret("CLICKHOUSE_PASSWORD", "")
        ch_db = get_secret("CLICKHOUSE_DB", "default")
        
        auth = f"{ch_user}:{ch_pass}@" if (ch_user or ch_pass) else ""
        ch_uri = f"clickhouse+http://{auth}{ch_host}:{ch_port}/{ch_db}"
        sanitized_target = f"clickhouse+http://{ch_host}:{ch_port}/{ch_db}"
        
        try:
            db = SQLDatabase.from_uri(ch_uri, include_tables=CLICKHOUSE_CORE_TABLES)
            db.get_table_info()  # Şema derleme doğrulaması
            logger.info(
                "Connected to ClickHouse successfully.",
                extra={"event_type": "database_connection", "dialect": "clickhouse", "host": ch_host}
            )
            return db, ch_uri, "clickhouse"
        except Exception as e:
            log_db_fallback(target_db=sanitized_target, fallback_db="sqlite:///insight_generation_bot.db", reason=str(e))

    # SQLite Yedek Veritabanı
    sqlite_uri = "sqlite:///insight_generation_bot.db"
    return SQLDatabase.from_uri(sqlite_uri), sqlite_uri, "sqlite"


def get_hybrid_agent():
    # --- 2. SQL BAĞLANTISI (ClickHouse -> SQLite Yedekli & Loglu) ---
    db, db_uri, dialect = get_database_connection()
    
    # --- 3. YAPAY ZEKA BEYNİ VE VEKTÖR MOTORU ---
    llm = ChatOpenAI(model=openai_model, temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- 4. RAG BAĞLANTISI (Metinler ve PDF'ler için) ---
    index_name = "pazarlama-verileri" 
    
    try:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        rag_tool = Tool(
            name="dokuman_arama_araci",
            description="Markanın iade politikaları, PDF raporları veya SQL veritabanında OLMAYAN yapılandırılmamış (unstructured) metinleri araştırmak için bu aracı kullan.",
            func=retriever.invoke
        )
        ekstra_araclar = [rag_tool]
        logger.info(
            "RAG tool integrated successfully.",
            extra={"event_type": "rag_init", "status": "success", "index_name": index_name}
        )
    except Exception as e:
        logger.warning(
            f"RAG sistemine bağlanılamadı, sadece SQL aracı devrede: {e}",
            extra={"event_type": "rag_init", "status": "failed", "error": str(e)}
        )
        ekstra_araclar = []

    # --- 5. HİBRİT AJAN OLUŞTURMA (SQL + RAG) ---
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        extra_tools=ekstra_araclar,
        verbose=False
    )
    
    # --- 6. ALT AJANLAR (SPJQ Query Agent & Rewrite NL Agent) ---
    query_agent = QueryAgent(db_uri=db_uri, model_name=openai_model, dialect=dialect, db=db)
    rewrite_agent = RewriteNLAgent(model_name=openai_model)
    
    return db, llm, agent_executor, query_agent, rewrite_agent


db, llm, agent_executor, query_agent, rewrite_agent = get_hybrid_agent()
