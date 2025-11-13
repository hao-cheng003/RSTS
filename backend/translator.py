from google.cloud import translate_v3 as translate

class GoogleTranslator:
    def __init__(self):
        self.client = translate.TranslationServiceClient()
        self.parent = "projects/iso-vision-414302/locations/global"

        # 统一的映射表：前端值 → Google API 语言码
        self.lang_map = {
            "english": "en",
            "japanese": "ja",
            "french": "fr",
            "spanish": "es",
            "chinese": "zh-CN",

            # 兼容 target="ja" / "fr" / "es" 形式
            "en": "en",
            "ja": "ja",
            "fr": "fr",
            "es": "es",
            "zh": "zh-CN",
            "zh-cn": "zh-CN",
            "zh-tw": "zh-TW",
        }

    def normalize(self, code: str) -> str:
        if not code:
            return "en"
        code = code.lower()

        # exact match
        if code in self.lang_map:
            return self.lang_map[code]

        # try first 2 characters
        code2 = code[:2]
        if code2 in self.lang_map:
            return self.lang_map[code2]

        return "en"

    def translate(self, text: str, src: str, target: str) -> str:
        if not text:
            return ""

        # normalize target language
        tgt = self.normalize(target)

        response = self.client.translate_text(
            request={
                "parent": self.parent,
                "contents": [text],
                "mime_type": "text/plain",
                "target_language_code": tgt,
                # "source_language_code": "auto",  # optional, auto-detect
            }
        )

        return response.translations[0].translated_text
