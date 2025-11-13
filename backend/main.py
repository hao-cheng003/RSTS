import asyncio, json, os, uuid, subprocess, time, hashlib
from typing import Optional, List
import numpy as np
import torch, whisper, webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

SAMPLE_RATE = 16000
VAD_AGGR = 3
FRAME_MS = 20
FRAME_LEN = int(SAMPLE_RATE * FRAME_MS / 1000)
MIN_SPEECH_SECONDS = 0.8
SILENCE_TAIL_SECONDS = 0.7
MAX_SEGMENT_SECONDS = 15.0

PROMPT_TAIL_LEN = 360
WHISPER_MODEL_SIZE = "small"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[boot] device(before) = {DEVICE}")
try:
    _model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
except NotImplementedError:
    DEVICE = "cpu"
    _model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
print(f"[boot] device(after)  = {DEVICE}")

TRANSLATOR_IMPL = os.getenv("TRANSLATOR", "google").lower()
if TRANSLATOR_IMPL == "google":
    from translator import GoogleTranslator as TranslatorImpl
    translator = TranslatorImpl()
else:
    from translator_local import ArgosTranslator as TranslatorImpl
    translator = TranslatorImpl()
print(f"[boot] translator     = {TRANSLATOR_IMPL}")

from google.cloud import texttospeech as gtts
_TTS_CLIENT = gtts.TextToSpeechClient()
_TTS_CACHE_DIR = "/tmp/rsts_tts_cache"
os.makedirs(_TTS_CACHE_DIR, exist_ok=True)

def _pick_voice(lang: str) -> tuple[str, str]:
    l = (lang or "en").lower()
    if l.startswith("zh"): return ("cmn-CN", "cmn-CN-Standard-A") 
    if l.startswith("en"): return ("en-US", "en-US-Neural2-F") 
    if l.startswith("ja"): return ("ja-JP", "ja-JP-Neural2-B")
    if l.startswith("fr"): return ("fr-FR", "fr-FR-Neural2-C")
    if l.startswith("es"): return ("es-ES", "es-ES-Neural2-B")
    return ("en-US", "en-US-Neural2-F")

def _tts_cache_key(text: str, lang: str) -> str:
    h = hashlib.sha256((lang + "||" + text).encode("utf-8")).hexdigest()
    return os.path.join(_TTS_CACHE_DIR, f"{h}.wav")

def _synthesize_google_tts(text: str, lang: str) -> str:

    out_path = _tts_cache_key(text, lang)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return out_path

    language_code, voice_name = _pick_voice(lang)
    input_ = gtts.SynthesisInput(text=text) 
    voice = gtts.VoiceSelectionParams(
        language_code=language_code, name=voice_name,
        ssml_gender=gtts.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED
    )
    audio_cfg = gtts.AudioConfig(
        audio_encoding=gtts.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000
    )
    resp = _TTS_CLIENT.synthesize_speech(input=input_, voice=voice, audio_config=audio_cfg)
    with open(out_path, "wb") as f:
        f.write(resp.audio_content)
    return out_path

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "device": DEVICE, "model": WHISPER_MODEL_SIZE, "translator": TRANSLATOR_IMPL, "tts": "google"}

def pcm16_bytes_to_f32(b: bytes) -> np.ndarray:
    if not b: return np.zeros(0, dtype=np.float32)
    i16 = np.frombuffer(b, dtype=np.int16)
    return (i16.astype(np.float32) / 32768.0)

class VADSegmenter:
    def __init__(self, sample_rate: int):
        self.vad = webrtcvad.Vad(VAD_AGGR)
        self.sr = sample_rate
        self.frame_len = FRAME_LEN
        self.min_speech_frames = int(MIN_SPEECH_SECONDS / (FRAME_MS / 1000))
        self.tail_sil_frames = int(SILENCE_TAIL_SECONDS / (FRAME_MS / 1000))
        self.max_seg_frames = int(MAX_SEGMENT_SECONDS / (FRAME_MS / 1000))
        self._buf = bytearray()
        self._frames: List[bytes] = []
        self._voiced_frames = 0
        self._silence_tail = 0
        self._in_speech = False

    def add_audio_bytes(self, pcm_bytes: bytes) -> List[bytes]:
        out_segments: List[bytes] = []
        if not pcm_bytes: return out_segments
        self._buf.extend(pcm_bytes)
        frame_size_bytes = self.frame_len * 2
        while len(self._buf) >= frame_size_bytes:
            chunk = bytes(self._buf[:frame_size_bytes]); del self._buf[:frame_size_bytes]
            is_voiced = self.vad.is_speech(chunk, self.sr)
            if is_voiced:
                if not self._in_speech:
                    self._frames = []; self._voiced_frames = 0; self._silence_tail = 0; self._in_speech = True
                self._frames.append(chunk); self._voiced_frames += 1
                if self._voiced_frames >= self.max_seg_frames:
                    if self._voiced_frames >= self.min_speech_frames:
                        out_segments.append(b"".join(self._frames))
                    self._frames = []; self._voiced_frames = 0; self._silence_tail = 0; self._in_speech = False
            else:
                if self._in_speech:
                    self._silence_tail += 1
                    self._frames.append(chunk)
                    if self._silence_tail >= self.tail_sil_frames:
                        if self._voiced_frames >= self.min_speech_frames:
                            out_segments.append(b"".join(self._frames))
                        self._frames = []; self._voiced_frames = 0; self._silence_tail = 0; self._in_speech = False
        return out_segments

    def flush(self) -> Optional[bytes]:
        if self._in_speech and self._voiced_frames >= self.min_speech_frames:
            seg = b"".join(self._frames)
            self._frames = []; self._voiced_frames = 0; self._silence_tail = 0; self._in_speech = False
            return seg
        return None

async def transcribe_once(audio_f32: np.ndarray, language: Optional[str], initial_prompt: str) -> str:
    opts = {
        "task": "transcribe",
        "language": language,                 # None=auto
        "fp16": (DEVICE == "cuda"),
        "temperature": 0.2,
        "beam_size": 2,
        "no_speech_threshold": 0.4,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": False,
        "initial_prompt": initial_prompt,
    }
    result = await asyncio.to_thread(_model.transcribe, audio_f32, **opts)
    return (result.get("text") or "").strip()

@app.post("/api/tts")
async def tts_endpoint(payload: dict, bg: BackgroundTasks):
    text = (payload.get("text") or "").strip()
    lang = (payload.get("lang") or "en").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    try:
        wav_path = _synthesize_google_tts(text, lang)
        out_path, mime = wav_path, "audio/wav"
    except Exception as e:
        return JSONResponse({"error": f"tts failed: {e}"}, status_code=500)
    return FileResponse(out_path, media_type=mime,
                        filename=os.path.basename(out_path),
                        headers={"Cache-Control": "no-store"})

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        cfg_text = await ws.receive_text()
        cfg = json.loads(cfg_text) if cfg_text else {}
        language = cfg.get("language", "auto")
        language = None if language == "auto" else language
        target = (cfg.get("target") or "en").strip()
        translate_enabled = bool(cfg.get("translate", True))
    except WebSocketDisconnect:
        return

    seg = VADSegmenter(SAMPLE_RATE)
    prompt_tail = ""

    async def handle_segment(pcm_bytes: bytes):
        nonlocal prompt_tail
        f32 = pcm16_bytes_to_f32(pcm_bytes)
        if f32.size < int(MIN_SPEECH_SECONDS * SAMPLE_RATE): return
        try:
            src = await transcribe_once(f32, language, prompt_tail)  # 左侧原文
        except Exception as e:
            await ws.send_text(json.dumps({"type":"error","message":str(e)})); return
        if not src: return
        tgt = translator.translate(src, language, target) if translate_enabled else ""
        await ws.send_text(json.dumps({"type":"final","src":src,"tgt":tgt}))
        prompt_tail = (prompt_tail + " " + src)[-PROMPT_TAIL_LEN:]

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect": break
            if "bytes" in msg and msg["bytes"] is not None:
                for s in seg.add_audio_bytes(msg["bytes"]): await handle_segment(s)
                continue
            if "text" in msg and msg["text"] is not None:
                txt = msg["text"].strip().lower()
                if txt == "close": break
                elif txt == "flush":
                    last = seg.flush()
                    if last: await handle_segment(last)
                continue
    except WebSocketDisconnect:
        pass
    finally:
        try:
            last = seg.flush()
            if last: await handle_segment(last)
        except: pass
        try:
            await ws.send_text(json.dumps({"type":"final","text":""}))
        except: pass

@app.get("/debug/translate")
def debug_translate(q: str = "Force a real translation call.", tgt: str = "zh"):
    out = translator.translate(q, None, tgt)
    return {"in": q, "out": out, "tgt": tgt}
