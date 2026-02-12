#!/usr/bin/env python3
"""Test simple de connexion Langfuse."""

import os
from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client

# Load env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

print("=== Langfuse Connection Test ===\n")

# Affiche les variables
secret = os.getenv("LANGFUSE_SECRET_KEY", "NOT SET")
public = os.getenv("LANGFUSE_PUBLIC_KEY", "NOT SET")
base_url = os.getenv("LANGFUSE_BASE_URL", "NOT SET")

print(f"LANGFUSE_SECRET_KEY: {secret[:20]}...")
print(f"LANGFUSE_PUBLIC_KEY: {public[:20]}...")
print(f"LANGFUSE_BASE_URL: {base_url}\n")

# Test connexion
langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")