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
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "")

# Use the proper endpoint - /transcribe path for the service
RUNPOD_ENDPOINT_URL = f"https://{RUNPOD_ENDPOINT_ID}.api.runpod.ai/transcribe"
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
LANGFUSE_BASE_URL=os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")


# Initialisation Langfuse
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)


def transcribe_audio(audio_file_path, language="auto", user_id=None):
    """
    Transcrit un fichier audio via RunPod avec tracking Langfuse
    """
    
    # Créer un trace_id
    trace_id = langfuse.create_trace_id()
    
    # Lire et encoder l'audio en base64
    with open(audio_file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
    
    # Extraire le sample_rate du fichier audio
    import wave
    try:
        with wave.open(audio_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
    except:
        sample_rate = 16000  # Default fallback
    
    # Préparer le payload - match the transcribe API format
    payload = {
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
    }
    
    # Démarrer un span pour l'appel API
    api_span = langfuse.start_span(
        name="runpod-api-call",
        trace_context={"trace_id": trace_id},
        input={
            "endpoint": RUNPOD_ENDPOINT_ID,
            "language": language,
            "audio_size_kb": len(audio_base64) / 1024,
            "audio_base64": audio_base64[:100] + "..."  # Garder juste les premiers 100 caractères pour la lisibilité
        },
        metadata={
            "audio_file": audio_file_path,
            "user_id": user_id,
            "sample_rate": sample_rate
        }
    )
    
    start_time = time.time()
    
    try:
        # Appel à RunPod
        response = requests.post(
            RUNPOD_ENDPOINT_URL,
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
        
        # Extraire les résultats directement du response
        transcription = result.get("transcription", "")
        detected_language = result.get("language", language)
        audio_duration = result.get("duration", 0)
        
        # Finaliser le span
        api_span.end()
        
        # Créer une génération pour le modèle ASR avec input/output complets
        generation = langfuse.start_generation(
            name="speech-to-text",
            trace_context={"trace_id": trace_id},
            model="whisper",
            input={
                "audio_base64": audio_base64,
                "audio_duration_seconds": audio_duration,
                "language": language,
                "sample_rate": sample_rate
            },
            metadata={
                "processing_time_ms": processing_time * 1000,
                "real_time_factor": processing_time / audio_duration if audio_duration > 0 else None,
                "audio_file": audio_file_path
            }
        )
        
        # Mettre à jour et finaliser la génération avec l'output
        generation.update(
            output=transcription
        )
        generation.end()
        
        # Mettre à jour la trace avec metadata
        langfuse.update_current_trace(
            user_id=user_id,
            metadata={
                "audio_file": audio_file_path,
                "endpoint_id": RUNPOD_ENDPOINT_ID
            },
            tags=["asr", "runpod-serverless", detected_language]
        )
        
        # Flush Langfuse
        langfuse.flush()
        
        return {
            "transcription": transcription,
            "language": detected_language,
            "duration": audio_duration,
            "processing_time": processing_time,
            "trace_id": trace_id
        }
        
    except Exception as e:
        # Finaliser le span avec erreur
        api_span.end()
        
        langfuse.flush()
        raise


# Utilisation
if __name__ == "__main__":
    result = transcribe_audio(
        audio_file_path="../output.wav",
        language="fr",
        user_id="user_123"
    )
    
    print(f"Transcription: {result['transcription']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Trace ID: {result['trace_id']}")