from typing import Optional
from translator import Translator

try:
    import argostranslate.translate as _arg_translate
    _ARGOS_OK = True
except Exception:
    _ARGOS_OK = False

class ArgosTranslator(Translator):
    def __init__(self):
        if not _ARGOS_OK:
            print("[ArgosTranslator] argostranslate not installed; will echo input.")

    def translate(self, text: str, src_lang: Optional[str], tgt_lang: str) -> str:
        if not text:
            return ""
        if not _ARGOS_OK:
            return text
        try:
            s = (src_lang or "").split("-")[0] if src_lang else None
            t = tgt_lang.split("-")[0]
            return _arg_translate.translate(text, s if s else None, t)
        except Exception:
            return text
