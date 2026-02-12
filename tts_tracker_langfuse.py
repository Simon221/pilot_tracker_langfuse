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
RUNPOD_TTS_ENDPOINT_ID = os.getenv("RUNPOD_TTS_ENDPOINT_ID", "")
RUNPOD_TTS_ENDPOINT_URL = os.getenv("RUNPOD_TTS_ENDPOINT_URL", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "")


# Initialisation Langfuse
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)


def synthesize_speech(text, language="auto", user_id=None):
    """
    Synthétise du texte en audio via RunPod avec tracking Langfuse
    """
    
    # Créer un trace_id
    trace_id = langfuse.create_trace_id()
    
    # Préparer le payload
    payload = {
        "text": text,
    }
    
    # Créer une génération pour le modèle TTS AVANT l'appel API
    generation = langfuse.start_generation(
        name="VOICEBOT_TTS",
        trace_context={"trace_id": trace_id},
        model="khady/snt_tts_voicebot_wavenet",
        input={
            "text": text,
            "language": language
        },
        metadata={
            "user_id": user_id
        }
    )
    
    start_time = time.time()
    
    try:
        # Appel à RunPod
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
        
        # Extraire l'audio en base64 du response
        audio_base64 = result.get("audio", "")
        
        # Mettre à jour la génération avec les résultats et métadatas
        generation.update(
            output={
                "audio": f"data:audio/wav;base64,{audio_base64}",
                "audio_length": len(audio_base64)
            },
            metadata={
                "processing_time_ms": processing_time * 1000,
                "user_id": user_id,
                "audio_size_bytes": len(audio_base64)
            }
        )
        generation.end()
        
        # Mettre à jour la trace avec metadata
        langfuse.update_current_trace(
            user_id=user_id,
            metadata={
                "endpoint_id": RUNPOD_TTS_ENDPOINT_ID,
                "text_length": len(text)
            },
            tags=["tts", "runpod-serverless", language]
        )
        
        # Flush Langfuse
        langfuse.flush()
        
        return {
            "audio_base64": audio_base64,
            "processing_time": processing_time,
            "trace_id": trace_id
        }
        
    except Exception as e:
        langfuse.flush()
        raise


# Utilisation
if __name__ == "__main__":
    result = synthesize_speech(
        text="Salam naka nga deif",
        language="fr",
        user_id="user_123"
    )
    
    print(f"Audio generated successfully")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Trace ID: {result['trace_id']}")
    print(f"Audio size: {len(result['audio_base64'])} bytes")
