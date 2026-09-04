import os
from typing import List, Tuple, Optional, Any
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st


class RewriteNLAgent:
    """
    Doğal Dil Yeniden Yazma ve Ayrıştırma Ajanı (Rewrite NL Agent).
    Kullanıcının makro iş sorularını, hipotezlerini ve tahminleme taleplerini
    veritabanı şemasına uygun atomik alt sorulara ve test edilebilir ifadelere dönüştürür.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0, llm: Optional[Any] = None):
        self.model_name = model_name
        self.temperature = temperature
        self._llm = llm

    @property
    def llm(self):
        """Lazy-loaded ChatOpenAI modeli."""
        if self._llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                try:
                    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                        api_key = st.secrets["OPENAI_API_KEY"]
                        os.environ["OPENAI_API_KEY"] = api_key
                except Exception:
                    pass
            self._llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        return self._llm

    def generate_macro_question(self, database_summary_info: str) -> str:
        """
        Otonom Mod için genel veritabanı özetinden vizyoner tek bir makro iş sorusu üretir.
        """
        hl_prompt = PromptTemplate.from_template(
            "Sen uzman bir pazarlama direktörüsün. Veritabanı özeti: {info}\n"
            "Lütfen marka sağlığını analiz etmek için vizyoner tek bir iş sorusu üret. Sadece soruyu yaz."
        )
        return (hl_prompt | self.llm).invoke({"info": database_summary_info}).content.strip()

    def decompose_question(self, macro_question: str, schema: str) -> Tuple[str, List[str]]:
        """
        Makro iş sorusunu veritabanı şemasına göre 2 net alt soruya ayrıştırır (SQL ve RAG rotası).
        """
        ll_prompt = PromptTemplate.from_template(
            "Sen kıdemli bir veri analistisin. Veritabanının şeması:\n{schema}\n\n"
            "Stratejik soru: {question}\n\n"
            "GÖREVİN: Bu soruyu çözmek için ajana rehberlik edecek 2 net alt soru kurgula.\n"
            "BİLGİ YÖNLENDİRMESİ:\n"
            "1. Eğer soru sayılar, oranlar, demografi veya duygularla ilgiliyse bunu ŞEMADAKİ sütunlara göre SQL sorusuna çevir.\n"
            "2. Eğer soru şirket politikaları, vizyon metinleri veya uzun dokümanlarla ilgiliyse bunu 'dokuman_arama_araci' ile çözülecek bir soruya çevir.\n"
            "3. KRİTİK: Soruların başına mutlaka tire (-) işareti koyarak liste halinde yaz."
        )
        raw_text = (ll_prompt | self.llm).invoke({"question": macro_question, "schema": schema}).content
        sub_questions = [
            line.lstrip("-* ").strip()
            for line in raw_text.split('\n')
            if line.strip().startswith('-') and line.lstrip("-* ").strip()
        ]
        return raw_text, sub_questions

    def formulate_hypothesis(self, topic: str, schema: str) -> Tuple[str, List[str]]:
        """
        Araştırma konusunu şemayı baz alan bir Alternatif Hipoteze (H1) ve 2 somut test sorusuna dönüştürür.
        """
        hyp_prompt = PromptTemplate.from_template(
            "Sen kıdemli bir veri bilimcisisin. Veritabanı şeması:\n{schema}\n\n"
            "Araştırma Konusu: {question}\n\n"
            "GÖREVİN: Bu konuyu test etmek için şemadaki sütunları baz alan tek bir Alternatif Hipotez (H1) üretmek "
            "ve SQL ajanının test edeceği 2 somut alt soru kurgulamak.\n\n"
            "ÇOK ÖNEMLİ KURALLAR:\n"
            "1. Aradığın veriler farklı tablolardaysa 'Tabloları JOIN yaparak birleştirin' şeklinde açık talimat ekle.\n"
            "2. Sorular kesinlikle matematiksel (COUNT, MAX, AVG) olsun. Ham tweet metni çekme (LIMIT hatası almamak için).\n"
            "3. Hipotezini 'EN ÇOK' gibi kesinleyici kelimeler yerine, daha esnek istatistiksel kavramlar üzerine kur.\n\n"
            "FORMAT KURALI:\n"
            "Hipotez (H1): [Hipotez cümlesi]\n"
            "- [1. net SQL sorusu]\n"
            "- [2. net SQL sorusu]"
        )
        raw_text = (hyp_prompt | self.llm).invoke({"question": topic, "schema": schema}).content
        sub_questions = [
            line.lstrip("-* ").strip()
            for line in raw_text.split('\n')
            if line.strip().startswith('-') and line.lstrip("-* ").strip()
        ]
        return raw_text, sub_questions

    def decompose_predictive_trends(self, topic: str, schema: str) -> Tuple[str, List[str]]:
        """
        Tahminleme talebini geçmiş trendleri yakalayacak 2 net zaman serisi / trend SQL sorusuna ayrıştırır.
        """
        pred_prompt = PromptTemplate.from_template(
            "Sen bir tahminleme (predictive) veri bilimcisisin. Veritabanı şeması:\n{schema}\n\n"
            "Kullanıcının Tahmin Talebi: {question}\n\n"
            "GÖREVİN: Geleceği tahmin edebilmemiz için bize GEÇMİŞ TRENDLERİ verecek 2 net SQL alt sorusu kurgulamak.\n"
            "KURALLAR:\n"
            "1. Zaman (date, timestamp, month vb.) sütunları varsa mutlaka onlara göre grupla (GROUP BY).\n"
            "2. Eğer zaman sütunu yoksa, veriyi büyüklük veya kategori bazında sıralayarak (ORDER BY) bir trend yakalamaya çalış.\n"
            "3. Soruların başına tire (-) koyarak liste halinde ver."
        )
        raw_text = (pred_prompt | self.llm).invoke({"question": topic, "schema": schema}).content
        sub_questions = [
            line.lstrip("-* ").strip()
            for line in raw_text.split('\n')
            if line.strip().startswith('-') and line.lstrip("-* ").strip()
        ]
        return raw_text, sub_questions


def get_rewrite_agent() -> RewriteNLAgent:
    """RewriteNLAgent için fabrika (factory) fonksiyonu."""
    return RewriteNLAgent()
