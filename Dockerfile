FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache AI models during build
RUN python -c "
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading embedding model...')
SentenceTransformer('intfloat/multilingual-e5-large')

print('Downloading LLM...')
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', torch_dtype=torch.float16)

print('Downloading ASR model...')
pipeline('automatic-speech-recognition', model='openai/whisper-base')

print('Downloading TTS model...')
pipeline('text-to-speech', model='facebook/mms-tts-kin')

print('All models cached successfully!')
"

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p legal_docs vector_db

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
