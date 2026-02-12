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
            n_frames = wav_file.getnframes()
            audio_duration = n_frames / sample_rate  # Calculer la durée en secondes
    except:
        sample_rate = 16000  # Default fallback
        audio_duration = 0
    
    # Préparer le payload - match the transcribe API format
    payload = {
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
    }
    
    # Créer une génération pour le modèle ASR AVANT l'appel API
    # Pour écouter l'audio dans Langfuse, on utilise le format data URI avec base64
    generation = langfuse.start_generation(
        name="speech-to-text",
        trace_context={"trace_id": trace_id},
        model="khady/snt_speech_to_text_fasterwhisper",
        input={
            "audio": f"data:audio/wav;base64,{audio_base64}",
            "language": language,
            "sample_rate": sample_rate
        },
        metadata={
            "audio_file_path": audio_file_path,
            "user_id": user_id,
            "audio_size_bytes": len(audio_base64)
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
        
        # Mettre à jour la génération avec les résultats et métadatas
        generation.update(
            output=transcription,
            metadata={
                "processing_time_ms": processing_time * 1000,
                "real_time_factor": processing_time / audio_duration if audio_duration > 0 else None,
                "audio_file": audio_file_path,
                "user_id": user_id,
                "audio_duration_seconds": audio_duration
            }
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