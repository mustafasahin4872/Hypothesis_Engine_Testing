"""
Sorgu Planlama Ajanı (Query Planning Agent)
Doğal dil sorularını yapılandırılmış JSON formatına (SPJQ) dönüştürür ve
deterministik SQL derleyicisi (agents.sql_compiler) üzerinden çalıştırır.
"""

import os
import json
import logging
import re
from typing import Any, Optional
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

from agents.sql_compiler import compile_json_to_sql
logger = logging.getLogger(__name__)

CLICKHOUSE_CORE_TABLES = ["tweet_predictions", "tweets", "users", "user_factors"]

# --- 1. API ANAHTARLARI (Streamlit Secrets & Ortam Değişkenleri ile Uyumlu) ---
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# --- 2. AJAN SİSTEM PROMPTU ---
_QUERY_GENERATOR_SYSTEM_PROMPT = """\
Sen uzman bir SQL ve JSON Sorgu Planlama Ajanısın (Query Planning Agent).
Görevin: Kullanıcının doğal dilde sorduğu iş veya pazarlama sorusunu inceleyerek, verilen veritabanı şemasına uygun tek tablolu yapılandırılmış (SPJQ) bir JSON sorgu nesnesine dönüştürmektir.

VERİTABANI ŞEMASI:
{schema}

BEKLENEN JSON ÇIKTI FORMATI:
{{
  "table": "<tablo_adi>",
  "columns": ["<sutun1>", "<sutun2>"],
  "aggregates": [
    {{"op": "count" | "count_distinct" | "sum" | "avg" | "min" | "max" | "group_array", "column": "<sutun_adi>", "as": "<takma_ad>"}}
  ],
  "filters": [
    {{"column": "<sutun_adi>", "op": "EQ" | "NEQ" | "GT" | "GTE" | "LT" | "LTE" | "LIKE" | "ILIKE" | "IN" | "NOT_IN" | "BETWEEN" | "HAS" | "HAS_ANY" | "HAS_ALL" | "IS_NULL" | "IS_NOT_NULL", "value": <deger>}}
  ],
  "group_by": ["<sutun1>"],
  "order_by": [
    {{"column": "<sutun_veya_takma_ad>", "dir": "asc" | "desc"}}
  ],
  "limit": <sayi>
}}

KRİTİK KURALLAR (HATA YAPMAMAK İÇİN MUTLAKA UY):
1. SÜTUN-TABLO EŞLEŞMESİ (EN ÖNEMLİ KURAL):
   - Bir sorguda 'table' olarak seçtiğin tablonun sütunları DIŞINDA HİÇBİR SÜTUNU o tablonun 'columns', 'aggregates' veya 'group_by' alanlarına YAZAMAZSIN!
   - 'tweets' tablosunun sütunları: id, text, created_at, impression_count, like_count, quote_count, reply_count, retweet_count, author_id, lang, created_year.
     * 'impression_count', 'like_count', 'retweet_count', 'text' SADECE 'tweets' tablosunda mevcuttur. ASLA 'users' tablosunda sum(impression_count) veya groupArray(text) yapma!
   - 'users' tablosunun sütunları: id, name, screen_name, gender, age_range, country_code, location, followers_count, following_count.
     * 'age_range', 'gender', 'country_code', 'location' SADECE 'users' tablosunda mevcuttur. ASLA 'tweets' tablosunda group_by: ['age_range'] yapma!
   - 'tweet_predictions' tablosunun sütunları: id, tweet_id, author_id, task_name, category_value, model_name.

2. DOĞRU TABLO SEÇİMİ REHBERİ:
   A) 'Hangi içerikler / tweetler daha fazla etkileşim alıyor?' sorusu için:
      - Ana tablo: 'tweets'
      - columns / group_by: ['id', 'text']
      - aggregates: [{{"op": "sum", "column": "impression_count", "as": "total_impressions"}}, {{"op": "sum", "column": "like_count", "as": "total_likes"}}]
      - order_by: [{{"column": "total_impressions", "dir": "desc"}}]
      - (Eğer hedef kitle/demografi filtresi varsa): filters: [{{"column": "author_id", "op": "IN", "value": {{"table": "users", "columns": ["id"], "filters": [{{"column": "age_range", "op": "IS_NOT_NULL"}}]}}}}]

   B) 'Hangi demografik gruplar (yaş, cinsiyet) daha çok şikayet ediyor / kullanıcı sayısı nedir?' sorusu için:
      - Ana tablo: 'users'
      - columns / group_by: ['age_range', 'gender']
      - aggregates: [{{"op": "count", "column": "id", "as": "user_count"}}]
      - filters: [{{"column": "id", "op": "IN", "value": {{"table": "tweet_predictions", "columns": ["author_id"], "filters": [{{"column": "task_name", "op": "EQ", "value": "consumer_journey"}}, {{"column": "category_value", "op": "HAS", "value": "Complaint"}}]}}}}]

3. Filtreleme (filters) kuralları:
   - EQ, NEQ, GT, GTE, LT, LTE, LIKE, ILIKE için 'value' tek bir metin veya sayı olmalıdır. Metinlerin başına/sonuna elle '%' işareti KOYMA.
   - IN / NOT_IN için 'value' bir liste olmalıdır (Örn: ["18-29", "30-39"]) veya alt tablo sorgu nesnesi (nested subquery).
   - ClickHouse dizi sütunları (category_value vb.) içeren sorgularda 'HAS' veya 'HAS_ANY' operatörlerini kullan.
   - BETWEEN için 'value' iki elemanlı bir liste olmalıdır [alt_sinir, ust_sinir].
   - IS_NULL / IS_NOT_NULL için 'value' boş bırakılabilir (null).
4. Sıralama veya en çok/en az sorularında 'order_by' ve 'limit' alanlarını doldur. Belirtilmemişse varsayılan limit 50'dir.
5. ÇIKTI KURALI: Sadece saf JSON formatında yanıt döndür. Markdown kod blokları (```json ... ```) veya açıklama metni YAZMA.
"""


# --- 3. SORGU PLANLAMA AJANI SINIFI (Lazy-Loaded LLM & DB) ---

class QueryAgent:
    """
    Doğal dil sorularını yapılandırılmış JSON formatına çeviren ve deterministik SQL üreten sorgu ajanı.
    """
    def __init__(
        self,
        db_uri: str = "sqlite:///insight_generation_bot.db",
        model_name: str = "gpt-4o",
        dialect: str = "sqlite",
        api_key: Optional[str] = None,
        llm: Optional[Any] = None,
        db: Optional[Any] = None,
        schema: Optional[str] = None
    ):
        self.db_uri = db_uri
        self.model_name = model_name
        self.dialect = dialect
        self.api_key = api_key
        self._db = db
        self._llm = llm
        self._schema = schema

    @property
    def db(self) -> SQLDatabase:
        if self._db is None:
            if self.dialect == "clickhouse" or "clickhouse" in self.db_uri:
                self._db = SQLDatabase.from_uri(self.db_uri, include_tables=CLICKHOUSE_CORE_TABLES)
            else:
                self._db = SQLDatabase.from_uri(self.db_uri)
        return self._db

    @property
    def schema(self) -> str:
        if self._schema is None:
            self._schema = self.db.get_table_info()
        return self._schema

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                try:
                    if "OPENAI_API_KEY" in st.secrets:
                        key = st.secrets["OPENAI_API_KEY"]
                        os.environ["OPENAI_API_KEY"] = key
                except Exception:
                    pass
            if not key:
                raise ValueError("OPENAI_API_KEY bulunamadı. Lütfen ortam değişkeni veya st.secrets üzerinden tanımlayın.")
            self._llm = ChatOpenAI(model=self.model_name, temperature=0, api_key=key)
        return self._llm

    def generate_query_json(self, question: str) -> dict[str, Any]:
        """
        Kullanıcı sorusunu ve veritabanı şemasını alarak LLM üzerinden yapılandırılmış JSON sorgusu üretir.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", _QUERY_GENERATOR_SYSTEM_PROMPT),
            ("user", "Bu soruyu yapılandırılmış JSON formatına çevir: {question}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"schema": self.schema, "question": question})
        raw_text = response.content.strip()

        # Markdown işaretlerini temizle
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text).strip()

        try:
            query_json = json.loads(raw_text)
            return query_json
        except json.JSONDecodeError as e:
            logger.error(f"LLM çıktısı JSON olarak ayrıştırılamadı: {raw_text}")
            raise ValueError(f"Geçersiz JSON formatı: {e}") from e

    def execute_nl_query(self, question: str, dialect: Optional[str] = None) -> dict[str, Any]:
        """
        Uçtan uca sorgu çalıştırma akışı:
        1. Doğal Dil -> Yapılandırılmış JSON (LLM)
        2. Yapılandırılmış JSON -> SQL (Deterministik Derleyici, dialect parametreli)
        3. SQL Çalıştırma -> Sonuçlar
        """
        active_dialect = dialect or self.dialect
        query_json = self.generate_query_json(question)
        sql = compile_json_to_sql(query_json, dialect=active_dialect)
        result = self.db.run(sql)
        return {
            "question": question,
            "json_query": query_json,
            "sql": sql,
            "result": result
        }


# Kolay erişim için fabrika fonksiyonu
def get_query_agent(db_uri: str = "sqlite:///insight_generation_bot.db", model_name: str = "gpt-4o", dialect: str = "sqlite"):
    return QueryAgent(db_uri=db_uri, model_name=model_name, dialect=dialect)
