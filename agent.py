import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import Tool
import streamlit as st
from agents.query_agent import QueryAgent, get_query_agent
from agents.rewrite_nl_agent import RewriteNLAgent, get_rewrite_agent

# --- 1. API ANAHTARLARI (GÜVENLİK BÖLGESİ - KASADAN ÇEKİLİYOR) ---
# Streamlit secrets veya ortam değişkenlerinden güvenli çekim
try:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if hasattr(st, "secrets") and "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
except Exception:
    pass


def get_hybrid_agent():
    # --- 2. SQL BAĞLANTISI (Sayılar ve Tablolar için) ---
    db_uri = "sqlite:///insight_generation_bot.db"
    db = SQLDatabase.from_uri(db_uri)
    
    # --- 3. YAPAY ZEKA BEYNİ VE VEKTÖR MOTORU ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
        print("✅ RAG Aracı başarıyla sisteme entegre edildi!")
    except Exception as e:
        print(f"⚠️ RAG sistemine bağlanılamadı, sadece SQL aracı devrede. Hata: {e}")
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
    query_agent = QueryAgent(db_uri=db_uri, model_name="gpt-4o")
    rewrite_agent = RewriteNLAgent(model_name="gpt-4o")
    
    return db, llm, agent_executor, query_agent, rewrite_agent


db, llm, agent_executor, query_agent, rewrite_agent = get_hybrid_agent()
