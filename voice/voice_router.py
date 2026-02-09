import io
import base64
import asyncio
import numpy as np
import soundfile as sf
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from scipy.signal import resample_poly
from dotenv import load_dotenv
from openai import OpenAI
from sarvamai import AsyncSarvamAI, AudioOutput
import os
import json
import requests
import tempfile
from pathlib import Path
import google.generativeai as genai


router = APIRouter(prefix="/voice", tags=["voice"])

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def ensure_voice_env():
    """
    Ensures required API keys for voice services are present.
    Called at request-time (NOT import-time).
    """
    missing = []

    if not SARVAM_API_KEY:
        missing.append("SARVAM_API_KEY")

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Voice services not configured",
                "missing_env_vars": missing,
            }
        )


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────── Helpers ────────────────────────────────

import subprocess
import tempfile
from pathlib import Path

def convert_to_wav(input_path: str) -> str:
    """
    Converts any audio format to WAV (mono, 16kHz, PCM 16-bit).
    Returns path to the converted WAV file.
    """
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz sample rate
        "-c:a", "pcm_s16le", # PCM 16-bit
        output.name
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output.name
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Audio conversion failed: {str(e)}")
    

async def gemini_translate_to_english(manipur_text: str) -> str:
    """
    Translates transcribed Manipuri text to English using Gemini Flash 3 model.
    Handles any Manipuri accent/dialect transcribed by Sarvam.
    """
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""You are a professional translator specializing in Manipuri languages and dialects.

Translate the following text from Manipuri (any dialect including Meiteilon, Tangkhul, Hmar, etc.) to English.
Provide only the English translation without any explanations, notes, or additional text.

Text to translate:
{manipur_text}

English translation:"""

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={
                'temperature': 1,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 8192,
            }
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini")
        
        return response.text.strip()
        
    except Exception as e:
        raise HTTPException(500, f"Gemini translation failed: {str(e)}")


def extract_audio(input_path: str) -> str:
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        # remove silence
        "-af", "silenceremove=start_periods=1:start_silence=0.5:start_threshold=-45dB",

        # mono + 16k
        "-ac", "1",
        "-ar", "16000",

        # raw pcm
        "-c:a", "pcm_s16le",

        out.name
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out.name

# async def sarvam_stt_manipuri(wav_bytes: bytes) -> str:
#     """
#     Transcribe Manipuri audio using Sarvam AI's batch job API.
#     Sarvam streaming doesn't support Manipuri, so we use their batch API.
#     """
#     import time
    
#     # Save wav_bytes to temp file for upload
#     temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     try:
#         temp_audio.write(wav_bytes)
#         temp_audio.close()
        
#         # Use synchronous client for batch job
#         from sarvamai import SarvamAI
#         client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        
#         # Create batch STT job for Manipuri
#         job = client.speech_to_text_job.create_job(
#             language_code="mni-IN",  # Manipuri
#             model="saaras:v3",
#             with_timestamps=False,
#             with_diarization=False
#         )
        
#         # Upload audio file
#         job.upload_files(file_paths=[temp_audio.name])
        
#         # Start processing
#         job.start()
        
#         # Wait for completion (max 60 seconds)
#         timeout = 60
#         start_time = time.time()
        
#         while time.time() - start_time < timeout:
#             status = job.get_status()
            
#             if job.is_complete():
#                 # Download results to temp directory
#                 output_dir = tempfile.mkdtemp()
#                 job.download_outputs(output_dir=output_dir)
                
#                 # Read transcription from output file
#                 # Sarvam outputs JSON files with transcription
#                 import glob
#                 json_files = glob.glob(f"{output_dir}/*.json")
                
#                 if not json_files:
#                     raise Exception("No transcription output received")
                
#                 with open(json_files[0], 'r', encoding='utf-8') as f:
#                     result = json.load(f)
                
#                 # Extract transcript from result
#                 transcript = result.get('transcript', '')
                
#                 # Cleanup output directory
#                 import shutil
#                 shutil.rmtree(output_dir, ignore_errors=True)
                
#                 if not transcript:
#                     raise Exception("Empty transcription received")
                    
#                 return transcript.strip()
                
#             elif job.is_failed():
#                 raise Exception(f"Sarvam job failed: {status}")
            
#             # Wait before checking again
#             await asyncio.sleep(2)
        
#         raise HTTPException(408, "Transcription timeout - job took too long")
        
#     except Exception as e:
#         raise HTTPException(500, f"Sarvam STT failed: {str(e)}")
#     finally:
#         # Cleanup temp audio file
#         try:
#             os.unlink(temp_audio.name)
#         except:
#             pass

async def sarvam_stt_manipuri(wav_path: str) -> str:
    """
    Transcribe Manipuri audio (any dialect) using Sarvam AI's batch job API.
    Sarvam will automatically detect the dialect and transcribe in Manipuri script.
    
    Args:
        wav_path: Path to WAV file (mono, 16kHz, PCM 16-bit)
    
    Returns:
        Transcribed text in Manipuri
    """
    import time
    import glob
    import shutil
    
    try:
        # Use synchronous client for batch job
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        
        # Create batch STT job for Manipuri
        job = client.speech_to_text_job.create_job(
            language_code="mni-IN",  # Manipuri - Sarvam auto-detects dialect
            model="saaras:v3",
            with_timestamps=False,
            with_diarization=False
        )
        
        # Upload audio file
        job.upload_files(file_paths=[wav_path])
        
        # Start processing
        job.start()
        
        # Wait for completion (adaptive timeout based on file size)
        file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        timeout = max(60, int(file_size_mb * 30))  # 30 seconds per MB, minimum 60s
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = job.get_status()
            
            if job.is_complete():
                # Download results to temp directory
                output_dir = tempfile.mkdtemp()
                job.download_outputs(output_dir=output_dir)
                
                # Read transcription from output file
                json_files = glob.glob(f"{output_dir}/*.json")
                
                if not json_files:
                    shutil.rmtree(output_dir, ignore_errors=True)
                    raise Exception("No transcription output received from Sarvam")
                
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                # Extract transcript from result
                transcript = result.get('transcript', '')
                
                # Cleanup output directory
                shutil.rmtree(output_dir, ignore_errors=True)
                
                if not transcript:
                    raise Exception("Empty transcription received from Sarvam")
                    
                return transcript.strip()
                
            elif job.is_failed():
                error_msg = f"Sarvam job failed: {status}"
                raise Exception(error_msg)
            
            # Wait before checking again
            await asyncio.sleep(2)
        
        raise HTTPException(408, f"Transcription timeout - job took longer than {timeout}s")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Sarvam STT failed: {str(e)}")



@router.post("/text-threat-check")
async def text_threat_check(text: str = Form(...)):

    ensure_voice_env()

    threat_result = detect_threat(text)

    return {
        "english_text": text,
        "threat_analysis": threat_result
    }






def detect_threat(text: str):
    """
    Uses OpenAI to classify threats.
    Pure prompt-based.
    """

    messages = [
        {
            "role": "system",
            "content": """
You are a security classifier.

Return ONLY valid JSON:

{
  "threat": true or false,
  "reason": "...",
  "severity": "low|medium|high"
}

Threats include bombs, weapons, shootings, attacks, violence.
"""
        },
        {"role": "user", "content": text}
    ]

    client = get_openai_client()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    return json.loads(resp.choices[0].message.content)



def normalize_wav_to_16k_mono(wav_bytes: bytes) -> bytes:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        up, down = 16000 // g, sr // g
        mono = resample_poly(mono, up, down)
    pcm16 = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
    out = io.BytesIO()
    sf.write(out, pcm16, 16000, subtype="PCM_16", format="WAV")
    return out.getvalue()


async def stt_transcribe(fixed_wav: bytes, lang="en-IN", timeout_s=20) -> str:
    client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    audio_b64 = base64.b64encode(fixed_wav).decode("utf-8")
    async with client.speech_to_text_streaming.connect(
        language_code=lang,
        model="saarika:v2.5",
        sample_rate=16000,
        input_audio_codec="wav",
        flush_signal=True,
    ) as ws:
        await ws.transcribe(audio=audio_b64, encoding="audio/wav", sample_rate=16000)
        await ws.flush()
        try:
            async with asyncio.timeout(timeout_s):
                async for msg in ws:
                    data = getattr(msg, "data", None)
                    if data and getattr(data, "transcript", None):
                        return data.transcript.strip()
        except asyncio.TimeoutError:
            return ""


def openai_chat(messages, model="gpt-4o-mini"):
    """✅ FIXED: Use the lazy-initialized client"""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return resp.choices[0].message.content.strip()


async def sarvam_tts(text: str, lang="en-IN", voice="anushka") -> bytes:
    client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    buf = io.BytesIO()
    async with client.text_to_speech_streaming.connect(model="bulbul:v2") as ws:
        await ws.configure(
            target_language_code=lang,
            speaker=voice,
            output_audio_codec="mp3",
            output_audio_bitrate="128k",
        )
        await ws.convert(text)
        await ws.flush()
        try:
            async with asyncio.timeout(10):
                async for msg in ws:
                    if isinstance(msg, AudioOutput):
                        buf.write(base64.b64decode(msg.data.audio))
        except asyncio.TimeoutError:
            pass
    return buf.getvalue()


# ─────────────────────────────── Endpoints ────────────────────────────────

@router.post("/record")
async def record_voice(
    file: UploadFile = File(...),
    stt_lang: str = Form("en-IN"),
    tts_lang: str = Form("en-IN"),
    tts_voice: str = Form("anushka"),
    llm_model: str = Form("gpt-4o-mini"),
):
    ensure_voice_env()
    raw = await file.read()
    fixed = normalize_wav_to_16k_mono(raw)
    transcript = await stt_transcribe(fixed, lang=stt_lang)
    if not transcript:
        raise HTTPException(400, "Failed to transcribe audio")

    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Keep replies concise."},
        {"role": "user", "content": transcript},
    ]
    reply = openai_chat(messages, model=llm_model)

    audio = await sarvam_tts(reply, lang=tts_lang, voice=tts_voice)
    return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")


@router.post("/stt")
async def stt_endpoint(file: UploadFile = File(...), lang: str = Form("en-IN")):

    ensure_voice_env()
    raw = await file.read()
    fixed = normalize_wav_to_16k_mono(raw)
    transcript = await stt_transcribe(fixed, lang=lang)
    return {"transcript": transcript}


@router.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    lang: str = Form("en-IN"),
    voice: str = Form("anushka")
):
    ensure_voice_env()

    audio = await sarvam_tts(text, lang=lang, voice=voice)
    return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")





# @router.post("/manipuri-threat-check")
# async def manipuri_threat_check(file: UploadFile = File(...)):

#     ensure_voice_env()

#     # 1. Save uploaded file
#     tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    
#     try:
#         content = await file.read()
#         with open(tmp_input.name, "wb") as f:
#             f.write(content)

#         # 2. Extract WAV (16k mono) using ffmpeg
#         wav_path = extract_audio(tmp_input.name)

#         # Validate audio file
#         if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 30_000:
#             raise HTTPException(400, "Audio too short or extraction failed")

#         # 3. Read WAV file
#         with open(wav_path, "rb") as f:
#             wav_bytes = f.read()

#         # Calculate duration
#         duration_seconds = len(wav_bytes) / 32000
        
#         manipuri_chunks = []
        
#         if duration_seconds > 25:
#             # LONG AUDIO: Split into 20-second chunks
#             chunk_duration = 20
#             num_chunks = int(np.ceil(duration_seconds / chunk_duration))
            
#             for i in range(num_chunks):
#                 chunk_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
#                 start_time = i * chunk_duration
                
#                 try:
#                     # Extract chunk using ffmpeg
#                     subprocess.run([
#                         "ffmpeg", "-y",
#                         "-i", wav_path,
#                         "-ss", str(start_time),
#                         "-t", str(chunk_duration),
#                         "-ac", "1",
#                         "-ar", "16000",
#                         "-c:a", "pcm_s16le",
#                         chunk_path
#                     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    
#                     if os.path.getsize(chunk_path) < 30_000:
#                         continue
                    
#                     # Read chunk
#                     with open(chunk_path, "rb") as cf:
#                         chunk_bytes = cf.read()
                    
#                     # Transcribe chunk with Sarvam
#                     try:
#                         chunk_text = await sarvam_stt_manipuri(chunk_bytes)
#                         if chunk_text and len(chunk_text) > 3:
#                             manipuri_chunks.append(chunk_text)
#                     except Exception as e:
#                         print(f"Chunk {i} transcription failed: {e}")
                        
#                 finally:
#                     # Cleanup chunk file
#                     try:
#                         if os.path.exists(chunk_path):
#                             os.unlink(chunk_path)
#                     except:
#                         pass
            
#             if not manipuri_chunks:
#                 raise HTTPException(400, "Failed to transcribe any audio chunks")
                
#             manipuri_text = " ".join(manipuri_chunks)
            
#         else:
#             # SHORT AUDIO: Process entire file
#             manipuri_text = await sarvam_stt_manipuri(wav_bytes)
        
#         if not manipuri_text or len(manipuri_text) < 5:
#             raise HTTPException(400, "Transcription too short or failed")
        
#         # Remove repetition artifacts from long audio
#         words = manipuri_text.split()
#         if len(words) > 30:
#             seen_phrases = {}
#             cleaned_words = []
            
#             for i in range(len(words)):
#                 if i + 5 <= len(words):
#                     phrase = " ".join(words[i:i+5])
#                     if phrase in seen_phrases:
#                         break
#                     seen_phrases[phrase] = True
#                 cleaned_words.append(words[i])
            
#             if len(cleaned_words) < len(words):
#                 manipuri_text = " ".join(cleaned_words)

#         # ============ TRANSLATION TO ENGLISH ============
        
#         # Sarvam provides transcription in native script or romanized
#         # We translate using OpenAI as a simple pass-through
#         client = get_openai_client()
        
#         translate_messages = [
#             {
#                 "role": "system",
#                 "content": """You are translating Manipuri (Meiteilon) text to English.

# Translate the text directly and literally. This is everyday speech - do NOT add poetic or metaphorical interpretations.

# Return ONLY the English translation."""
#             },
#             {
#                 "role": "user",
#                 "content": f"Translate to English:\n\n{manipuri_text}"
#             }
#         ]

#         english_text = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=translate_messages,
#             temperature=0.1
#         ).choices[0].message.content.strip()

#         if not english_text or len(english_text) < 3:
#             raise HTTPException(400, "Translation failed")

#         # ============ THREAT DETECTION ============
        
#         threat_result = detect_threat(english_text)

#         return {
#             "manipuri_text": manipuri_text,
#             "english_text": english_text,
#             "threat_analysis": threat_result,
#             "audio_duration": round(duration_seconds, 1),
#             "chunks_processed": len(manipuri_chunks) if duration_seconds > 25 else 1
#         }
        
#     finally:
#         # Cleanup all temp files
#         try:
#             if os.path.exists(tmp_input.name):
#                 os.unlink(tmp_input.name)
#         except Exception as e:
#             print(f"Cleanup error for input: {e}")
            
#         try:
#             if 'wav_path' in locals() and os.path.exists(wav_path):
#                 # Wait a bit before deleting to ensure file is released
#                 import time
#                 time.sleep(0.1)
#                 os.unlink(wav_path)
#         except Exception as e:
#             print(f"Cleanup error for wav: {e}")


@router.post("/manipuri-threat-check")
async def manipuri_threat_check(file: UploadFile = File(...)):
    """
    Two-stage pipeline:
    1. Sarvam AI: Audio (any Manipuri dialect) → Manipuri text transcription
    2. Gemini Flash 3: Manipuri text → English translation
    3. OpenAI: English text → Threat detection
    
    Handles any uploaded audio format by converting to WAV first.
    No chunking - processes entire audio file for best quality.
    """
    ensure_voice_env()
    
    temp_input = None
    wav_path = None
    
    try:
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
        content = await file.read()
        temp_input.write(content)
        temp_input.close()
        
        # Convert to WAV format (mono, 16kHz, PCM 16-bit)
        wav_path = convert_to_wav(temp_input.name)
        
        # Stage 1: Sarvam transcription (Manipuri audio → Manipuri text)
        manipur_text = await sarvam_stt_manipuri(wav_path)
        
        if not manipur_text:
            raise HTTPException(400, "No transcription generated from audio")
        
        # Stage 2: Gemini translation (Manipuri text → English text)
        english_text = await gemini_translate_to_english(manipur_text)
        
        if not english_text:
            raise HTTPException(400, "Translation failed to produce output")
        
        # Stage 3: Threat detection on English text
        threat_result = detect_threat(english_text)
        
        return {
            "manipur_text": manipur_text,
            "english_text": english_text,
            "threat_analysis": threat_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        # Cleanup temporary files
        if temp_input:
            try:
                os.unlink(temp_input.name)
            except:
                pass
        if wav_path:
            try:
                os.unlink(wav_path)
            except:
                pass

@router.websocket("/stream")
async def voice_stream(websocket: WebSocket):

    try:
        ensure_voice_env()
    except HTTPException as e:
        await websocket.close(code=1011)
        return

    """
    Bidirectional voice streaming over WebSocket.

    Client protocol (JSON text messages unless noted):
    - {"type":"start", "stt_lang":"en-IN", "tts_lang":"en-IN", "tts_voice":"anushka", "llm_model":"gpt-5-mini"}
    - {"type":"chunk", "audio":"<base64>", "format":"wav"|"pcm16", "sample_rate":16000}
    - {"type":"flush"}

    Server messages:
    - {"type":"ready"}
    - {"type":"stt_partial", "text":"..."}
    - {"type":"stt_final", "text":"..."}
    - {"type":"assistant_text", "text":"..."}
    - {"type":"tts_chunk", "audio":"<base64 mp3>"}
    - {"type":"tts_done"}
    - {"type":"error", "message":"..."}
    """
    await websocket.accept()

    stt_lang = "en-IN"
    tts_lang = "en-IN"
    tts_voice = "anushka"
    llm_model = "gpt-5-mini"

    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Keep replies concise."}
    ]

    client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

    transcript_queue: asyncio.Queue[str] = asyncio.Queue()
    last_partial: str | None = None

    async def stt_reader(stt_ws):
        nonlocal last_partial
        try:
            async for msg in stt_ws:
                data = getattr(msg, "data", None)
                if data and getattr(data, "transcript", None):
                    text = (data.transcript or "").strip()
                    if text:
                        last_partial = text
                        await transcript_queue.put(text)
                        try:
                            await websocket.send_text(json.dumps({"type": "stt_partial", "text": text}))
                        except Exception:
                            pass
        except Exception:
            return

    try:
        # Optional start config
        start_buffer = None
        try:
            first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            try:
                obj = json.loads(first_msg)
            except Exception:
                obj = {}
            if isinstance(obj, dict) and obj.get("type") == "start":
                stt_lang = obj.get("stt_lang", stt_lang)
                tts_lang = obj.get("tts_lang", tts_lang)
                tts_voice = obj.get("tts_voice", tts_voice)
                llm_model = obj.get("llm_model", llm_model)
            else:
                start_buffer = first_msg
        except asyncio.TimeoutError:
            pass

        async with client.speech_to_text_streaming.connect(
            language_code=stt_lang,
            model="saarika:v2.5",
            sample_rate=16000,
            input_audio_codec="wav",
            flush_signal=True,
        ) as stt_ws:
            reader_task = asyncio.create_task(stt_reader(stt_ws))
            await websocket.send_text(json.dumps({"type": "ready"}))

            # Process buffered first message if any
            if start_buffer:
                try:
                    obj = json.loads(start_buffer)
                    if isinstance(obj, dict) and obj.get("type") == "chunk" and obj.get("audio"):
                        fmt = obj.get("format", "wav")
                        sr = int(obj.get("sample_rate", 16000))
                        if fmt not in {"wav", "pcm16"}:
                            fmt = "wav"
                        await stt_ws.transcribe(audio=obj["audio"], encoding=f"audio/{fmt}", sample_rate=sr)
                except Exception:
                    pass

            while True:
                try:
                    msg = await websocket.receive()
                except WebSocketDisconnect:
                    break

                if msg.get("type") == "websocket.receive":
                    if msg.get("text") is not None:
                        try:
                            obj = json.loads(msg["text"])
                        except Exception:
                            await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                            continue
                        mtype = obj.get("type")
                        if mtype == "chunk":
                            audio_b64 = obj.get("audio")
                            fmt = obj.get("format", "wav")
                            sr = int(obj.get("sample_rate", 16000))
                            if not audio_b64:
                                await websocket.send_text(json.dumps({"type": "error", "message": "Missing audio"}))
                                continue
                            if fmt not in {"wav", "pcm16"}:
                                fmt = "wav"
                            try:
                                await stt_ws.transcribe(audio=audio_b64, encoding=f"audio/{fmt}", sample_rate=sr)
                            except Exception as e:
                                await websocket.send_text(json.dumps({"type": "error", "message": f"STT chunk error: {e}"}))
                        elif mtype == "flush":
                            try:
                                await stt_ws.flush()
                            except Exception:
                                pass

                            collected = []
                            try:
                                while True:
                                    collected.append(
                                        transcript_queue.get_nowait())
                            except asyncio.QueueEmpty:
                                pass
                            try:
                                with asyncio.timeout(1.0):
                                    while True:
                                        t = await transcript_queue.get()
                                        collected.append(t)
                            except asyncio.TimeoutError:
                                pass

                            final_text = (
                                collected[-1] if collected else (last_partial or "")).strip()
                            await websocket.send_text(json.dumps({"type": "stt_final", "text": final_text}))

                            if final_text:
                                messages.append(
                                    {"role": "user", "content": final_text})
                                try:
                                    reply_text = openai_chat(
                                        messages, model=llm_model)
                                except Exception as e:
                                    await websocket.send_text(json.dumps({"type": "error", "message": f"LLM error: {e}"}))
                                    continue
                                messages.append(
                                    {"role": "assistant", "content": reply_text})
                                await websocket.send_text(json.dumps({"type": "assistant_text", "text": reply_text}))

                                # TTS stream reply
                                try:
                                    async with client.text_to_speech_streaming.connect(model="bulbul:v2") as tts_ws:
                                        await tts_ws.configure(
                                            target_language_code=tts_lang,
                                            speaker=tts_voice,
                                            output_audio_codec="mp3",
                                            output_audio_bitrate="128k",
                                        )
                                        await tts_ws.convert(reply_text)
                                        await tts_ws.flush()
                                        try:
                                            async with asyncio.timeout(10):
                                                async for tmsg in tts_ws:
                                                    if isinstance(tmsg, AudioOutput):
                                                        await websocket.send_text(json.dumps({"type": "tts_chunk", "audio": tmsg.data.audio}))
                                        except asyncio.TimeoutError:
                                            pass
                                except Exception as e:
                                    await websocket.send_text(json.dumps({"type": "error", "message": f"TTS error: {e}"}))
                                finally:
                                    await websocket.send_text(json.dumps({"type": "tts_done"}))
                        elif mtype == "start":
                            pass
                        else:
                            await websocket.send_text(json.dumps({"type": "error", "message": "Unknown message type"}))
                    elif msg.get("bytes") is not None:
                        try:
                            b64 = base64.b64encode(
                                msg["bytes"]).decode("utf-8")
                            await stt_ws.transcribe(audio=b64, encoding="audio/pcm16", sample_rate=16000)
                        except Exception as e:
                            await websocket.send_text(json.dumps({"type": "error", "message": f"STT binary chunk error: {e}"}))
                elif msg.get("type") == "websocket.disconnect":
                    break

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
