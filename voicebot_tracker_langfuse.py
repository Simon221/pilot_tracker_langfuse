from langfuse import Langfuse
import requests
import time
import os
import base64
from pathlib import Path
from dotenv import load_dotenv


# Load env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ASR_ENDPOINT_ID = os.getenv("RUNPOD_ASR_ENDPOINT_ID", "")
RUNPOD_TTS_ENDPOINT_ID = os.getenv("RUNPOD_TTS_ENDPOINT_ID", "")
RUNPOD_ASR_ENDPOINT_URL = f"https://{RUNPOD_ASR_ENDPOINT_ID}.api.runpod.ai/transcribe"
RUNPOD_TTS_ENDPOINT_URL = f"https://{RUNPOD_TTS_ENDPOINT_ID}.api.runpod.ai/synthesize"

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")


# Initialisation Langfuse
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)


def transcribe_audio(audio_file_path, language="auto"):
    """
    Transcrit un fichier audio via RunPod
    """
    
    # Lire et encoder l'audio en base64
    with open(audio_file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
    
    # Extraire le sample_rate du fichier audio
    import wave
    try:
        with wave.open(audio_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_duration = n_frames / sample_rate
    except:
        sample_rate = 16000
        audio_duration = 0
    
    # Préparer le payload
    payload = {
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
    }
    
    start_time = time.time()
    
    try:
        # Appel à RunPod ASR
        response = requests.post(
            RUNPOD_ASR_ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=300
        )
        
        response.raise_for_status()
        
        processing_time = time.time() - start_time
        result = response.json()
        
        # Extraire les résultats
        transcription = result.get("transcription", "")
        detected_language = result.get("language", language)
        
        return {
            "transcription": transcription,
            "language": detected_language,
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "audio_file_path": audio_file_path
        }
        
    except Exception as e:
        raise


def synthesize_speech(text, language="auto"):
    """
    Synthétise du texte en audio via RunPod
    """
    
    # Préparer le payload
    payload = {
        "text": text,
    }
    
    start_time = time.time()
    
    try:
        # Appel à RunPod TTS
        response = requests.post(
            RUNPOD_TTS_ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=300
        )
        
        response.raise_for_status()
        
        processing_time = time.time() - start_time
        result = response.json()
        
        # Extraire l'audio en base64
        audio_base64 = result.get("audio", "")
        
        return {
            "audio_base64": audio_base64,
            "processing_time": processing_time,
            "text_length": len(text)
        }
        
    except Exception as e:
        raise


def voicebot_pipeline(audio_file_path, language="auto", user_id=None):
    """
    Pipeline complet Voicebot: ASR -> Processing -> TTS avec tracking Langfuse
    """
    
    # Créer un trace_id pour toute la conversation
    trace_id = langfuse.create_trace_id()
    
    # Créer un span principal pour tout le pipeline Voicebot
    voicebot_span = langfuse.start_span(
        name="VOICEBOT_PIPELINE",
        trace_context={"trace_id": trace_id},
        input={
            "audio_file": audio_file_path,
            "language": language
        },
        metadata={
            "user_id": user_id,
            "pipeline_type": "asr_tts"
        }
    )
    
    try:
        # ===== ÉTAPE 1: ASR (Speech-to-Text) =====
        asr_start = time.time()
        asr_result = transcribe_audio(audio_file_path, language)
        asr_time = time.time() - asr_start
        
        # Créer une génération enfant pour l'ASR
        asr_generation = langfuse.start_generation(
            name="VOICEBOT_ASR",
            trace_context={"trace_id": trace_id, "parent_observation_id": voicebot_span.id},
            model="khady/snt_speech_to_text_fasterwhisper",
            input={
                "audio": f"data:audio/wav;base64,{asr_result['audio_base64']}",
                "language": language,
                "sample_rate": asr_result['sample_rate']
            },
            metadata={
                "audio_file": asr_result['audio_file_path'],
                "user_id": user_id,
                "audio_duration_seconds": asr_result['audio_duration']
            }
        )
        
        asr_generation.update(
            output=asr_result['transcription'],
            metadata={
                "processing_time_ms": asr_result['processing_time'] * 1000,
                "real_time_factor": asr_result['processing_time'] / asr_result['audio_duration'] if asr_result['audio_duration'] > 0 else None,
                "detected_language": asr_result['language']
            }
        )
        asr_generation.end()
        
        # ===== ÉTAPE 2: TTS (Text-to-Speech) =====
        tts_start = time.time()
        tts_result = synthesize_speech(asr_result['transcription'], asr_result['language'])
        tts_time = time.time() - tts_start
        
        # Créer une génération enfant pour le TTS
        tts_generation = langfuse.start_generation(
            name="VOICEBOT_TTS",
            trace_context={"trace_id": trace_id, "parent_observation_id": voicebot_span.id},
            model="khady/snt_tts_voicebot_wavenet",
            input={
                "text": asr_result['transcription'],
                "language": asr_result['language']
            },
            metadata={
                "user_id": user_id,
                "text_length": tts_result['text_length']
            }
        )
        
        tts_generation.update(
            output={
                "audio": f"data:audio/wav;base64,{tts_result['audio_base64']}",
                "audio_length": len(tts_result['audio_base64'])
            },
            metadata={
                "processing_time_ms": tts_result['processing_time'] * 1000,
                "audio_size_bytes": len(tts_result['audio_base64'])
            }
        )
        tts_generation.end()
        
        # ===== FINALISER LE PIPELINE =====
        total_time = asr_time + tts_time
        
        voicebot_span.update(
            output={
                "transcription": asr_result['transcription'],
                "audio": f"data:audio/wav;base64,{tts_result['audio_base64']}",
                "language": asr_result['language']
            },
            metadata={
                "asr_processing_time_ms": asr_result['processing_time'] * 1000,
                "tts_processing_time_ms": tts_result['processing_time'] * 1000,
                "total_processing_time_ms": total_time * 1000,
                "asr_endpoint_id": RUNPOD_ASR_ENDPOINT_ID,
                "tts_endpoint_id": RUNPOD_TTS_ENDPOINT_ID,
                "user_id": user_id
            }
        )
        voicebot_span.end()
        
        langfuse.flush()
        
        return {
            "transcription": asr_result['transcription'],
            "language": asr_result['language'],
            "audio_base64": tts_result['audio_base64'],
            "asr_processing_time": asr_result['processing_time'],
            "tts_processing_time": tts_result['processing_time'],
            "total_processing_time": total_time,
            "trace_id": trace_id
        }
        
    except Exception as e:
        voicebot_span.end()
        langfuse.flush()
        raise


# Utilisation
if __name__ == "__main__":
    result = voicebot_pipeline(
        audio_file_path="../output.wav",
        language="fr",
        user_id="user_123"
    )
    
    print(f"\n=== VOICEBOT PIPELINE RESULTS ===")
    print(f"Transcription: {result['transcription']}")
    print(f"Language: {result['language']}")
    print(f"ASR Processing time: {result['asr_processing_time']:.2f}s")
    print(f"TTS Processing time: {result['tts_processing_time']:.2f}s")
    print(f"Total Processing time: {result['total_processing_time']:.2f}s")
    print(f"Trace ID: {result['trace_id']}")
    print(f"Audio generated: {len(result['audio_base64'])} bytes")
