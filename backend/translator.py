from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional
import os
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, src_lang: Optional[str], tgt_lang: str) -> str:
        ...

    def translate_batch(self, texts: List[str], src_lang: Optional[str], tgt_lang: str) -> List[str]:
        return [self.translate(t, src_lang, tgt_lang) for t in texts]

class GoogleTranslator(Translator):
    def __init__(self, project_id: Optional[str] = None, location: Optional[str] = None):
        from google.cloud import translate
        self._translate = translate
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_LOCATION", "global")
        assert self.project_id, "GCP_PROJECT_ID is required"
        self.client = self._translate.TranslationServiceClient()
        self.parent = f"projects/{self.project_id}/locations/{self.location}"

    @retry(wait=wait_exponential_jitter(0.2, 2.0), stop=stop_after_attempt(3))
    def _call(self, text: str, src: Optional[str], tgt: str) -> str:
        req = {
            "parent": self.parent,
            "contents": [text],
            "mime_type": "text/plain",
            "target_language_code": tgt,
        }
        if src:
            req["source_language_code"] = src
        resp = self.client.translate_text(request=req)
        return resp.translations[0].translated_text if resp.translations else ""

    @lru_cache(maxsize=2048)
    def _cached(self, text: str, src: Optional[str], tgt: str) -> str:
        return self._call(text, src, tgt)

    def translate(self, text: str, src_lang: Optional[str], tgt_lang: str) -> str:
        if not text or not text.strip():
            return ""
        if len(text.strip()) < 4:
            return text
        return self._cached(text.strip(), (src_lang or "").strip() or None, tgt_lang.strip())
