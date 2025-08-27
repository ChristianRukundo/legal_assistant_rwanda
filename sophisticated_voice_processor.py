"""
Sophisticated Voice Processor with Advanced ASR, TTS, and Audio Intelligence
Handles multi-language voice processing with emotion detection and adaptive responses
"""

import asyncio
import io
import json
import numpy as np
import librosa
import soundfile as sf
import webrtcvad
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import tempfile
import hashlib
# Removed unused imports: from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# Removed unused import: import multiprocessing as mp

import torch
import torch.nn.functional as F
# Removed unused import: import torchaudio
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan,
    pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
)
# Removed unused imports: from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
import whisper
from TTS.api import TTS # This import is crucial and expected to work if TTS is installed
# Removed unused imports: import pyaudio, import wave
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import noisereduce as nr
import scipy.signal
# Removed unused import: from scipy.io import wavfile
import structlog

# Initialize logging
logger = structlog.get_logger(__name__)

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_size: int = 1024
    max_duration: int = 300  # 5 minutes
    min_duration: float = 0.5  # 0.5 seconds
    noise_reduction: bool = True
    voice_activity_detection: bool = True
    audio_enhancement: bool = True
    emotion_detection: bool = True
    speaker_identification: bool = True

@dataclass
class VoiceAnalysis:
    """Comprehensive voice analysis results"""
    transcription: str
    confidence_score: float
    detected_language: str
    language_confidence: float
    emotion_analysis: Dict[str, float] = field(default_factory=dict)
    speaker_characteristics: Dict[str, Any] = field(default_factory=dict)
    audio_quality: Dict[str, float] = field(default_factory=dict)
    speech_rate: float = 0.0
    pause_analysis: Dict[str, Any] = field(default_factory=dict)
    pronunciation_assessment: Dict[str, float] = field(default_factory=dict)

@dataclass
class TTSConfig:
    """Configuration for text-to-speech synthesis"""
    voice_model: str = "tts_models/multilingual/multi-dataset/xtts_v2" # This will download if not local
    speaking_rate: float = 1.0
    pitch_shift: float = 0.0
    emotion_style: str = "neutral"
    voice_clone: bool = False
    speaker_embedding: Optional[np.ndarray] = None
    language_specific_models: Dict[str, str] = field(default_factory=dict)

class AdvancedASREngine:
    """Advanced Automatic Speech Recognition with multi-model support"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.models = {}
        self.processors = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configurations for different languages and use cases
        self.model_configs = {
            "whisper_large": "large-v3", # Use whisper's built-in name for model loading
            "wav2vec2_multilingual": "facebook/wav2vec2-large-xlsr-53",
            "wav2vec2_english": "facebook/wav2vec2-large-960h-lv60-self",
            # "speechbrain_asr": "speechbrain/asr-wav2vec2-commonvoice-en", # Removed as SpeechBrain is not used
            "kinyarwanda_asr": "facebook/wav2vec2-large-xlsr-53"  # Fine-tuned for Kinyarwanda, assumes specific fine-tuning or zero-shot capability
        }
        self.whisper_model: Any = None # Initialize as None
        self.language_detector: Any = None # Initialize as None
    
    async def initialize(self):
        """Initialize all ASR models"""
        logger.info("Initializing Advanced ASR Engine...")
        
        try:
            # Load Whisper model (primary for multilingual)
            logger.info(f"Loading Whisper model: {self.model_configs['whisper_large']} on {self.device}")
            self.whisper_model = whisper.load_model(self.model_configs['whisper_large'], device=self.device)
            logger.info("Whisper model loaded.")
            
            # Load Wav2Vec2 models
            await self._load_wav2vec2_models()
            
            # Load language detection pipeline
            logger.info("Loading language detection model...")
            self.language_detector = pipeline(
                "audio-classification",
                model="facebook/wav2vec2-large-xlsr-53", # Example model for language ID
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Language detection model loaded.")
            
            logger.info("ASR Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ASR Engine: {e}", exc_info=True)
            raise
    
    async def _load_wav2vec2_models(self):
        """Load Wav2Vec2 models for different languages"""
        for model_name, model_path in self.model_configs.items():
            if "wav2vec2" in model_name or "kinyarwanda_asr" in model_name: # Ensure Kinyarwanda ASR is also handled
                try:
                    logger.info(f"Loading Wav2Vec2 model: {model_path} for {model_name}")
                    processor = Wav2Vec2Processor.from_pretrained(model_path)
                    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
                    
                    self.processors[model_name] = processor
                    self.models[model_name] = model
                    
                    logger.info(f"Loaded {model_name} model successfully.")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} from {model_path}: {e}")
                    # Remove from config if it failed to load
                    self.model_configs.pop(model_name, None)

    
    async def transcribe_audio(self, audio_data: bytes, expected_language: str = "auto") -> VoiceAnalysis:
        """Comprehensive audio transcription with analysis"""
        if self.whisper_model is None:
            raise RuntimeError("ASR Engine not initialized. Whisper model is not loaded.")

        try:
            # Preprocess audio
            processed_audio = await self._preprocess_audio(audio_data)
            
            # Detect language if not specified or fallback
            detected_language = expected_language
            if expected_language == "auto":
                detected_language = await self._detect_language(processed_audio)
            
            # Choose best model for language
            best_model_name = self._select_best_model(detected_language)
            
            # Perform transcription
            transcription_result = await self._transcribe_with_model(processed_audio, best_model_name, detected_language)
            
            # Analyze audio characteristics
            audio_analysis = await self._analyze_audio_characteristics(processed_audio)
            
            # Combine results
            voice_analysis = VoiceAnalysis(
                transcription=transcription_result["text"],
                confidence_score=transcription_result["confidence"],
                detected_language=detected_language,
                language_confidence=transcription_result.get("language_confidence", 0.8),
                emotion_analysis=audio_analysis.get("emotions", {}),
                speaker_characteristics=audio_analysis.get("speaker", {}),
                audio_quality=audio_analysis.get("quality", {}),
                speech_rate=audio_analysis.get("speech_rate", 0.0),
                pause_analysis=audio_analysis.get("pauses", {}),
                pronunciation_assessment=audio_analysis.get("pronunciation", {})
            )
            
            return voice_analysis
            
        except Exception as e:
            logger.error(f"Error in audio transcription: {e}", exc_info=True)
            raise
    
    async def _preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Advanced audio preprocessing pipeline"""
        # Convert bytes to audio array
        # pydub.AudioSegment handles various formats
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
        
        # Convert to mono and resample
        audio_segment = audio_segment.set_channels(1).set_frame_rate(self.config.sample_rate)
        
        # Convert to numpy array (pydub stores in bytes, convert to float32)
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize to -1 to 1
        
        # Apply noise reduction
        if self.config.noise_reduction:
            audio_array = nr.reduce_noise(y=audio_array, sr=self.config.sample_rate)
        
        # Apply audio enhancement
        if self.config.audio_enhancement:
            audio_array = self._enhance_audio(audio_array)
        
        # Voice activity detection
        if self.config.voice_activity_detection:
            audio_array = self._apply_vad(audio_array)
        
        return audio_array
    
    def _enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio enhancement techniques"""
        # Apply high-pass filter to remove low-frequency noise
        sos = scipy.signal.butter(5, 80, btype='high', fs=self.config.sample_rate, output='sos')
        audio = scipy.signal.sosfilt(sos, audio)
        
        # Apply dynamic range compression
        audio_segment = AudioSegment(
            audio.tobytes(), 
            frame_rate=self.config.sample_rate,
            sample_width=audio.dtype.itemsize, # Use actual itemsize for sample_width
            channels=1
        )
        compressed = compress_dynamic_range(audio_segment, threshold=-20.0, ratio=4.0)
        
        # Convert back to numpy array
        # Ensure conversion from pydub's internal format back to float32
        enhanced_audio = np.array(compressed.get_array_of_samples(), dtype=np.float32)
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio))
        
        return enhanced_audio
    
    def _apply_vad(self, audio: np.ndarray) -> np.ndarray:
        """Apply Voice Activity Detection to remove silence"""
        # webrtcvad operates on 16-bit PCM samples
        audio_16bit = (audio * 32767).astype(np.int16)
        
        # Create VAD instance
        # Aggressiveness mode: 0 (least aggressive) to 3 (most aggressive)
        # Higher aggressiveness means more frames are considered non-speech.
        vad = webrtcvad.Vad(2)
        
        # Process audio in 10, 20, or 30 ms frames
        frame_duration_ms = 30 # ms
        frame_size = int(self.config.sample_rate * frame_duration_ms / 1000)
        
        # Ensure frame_size is valid for VAD (usually must be 160, 320, or 480 for 8kHz, 16kHz, 32kHz)
        # For 16kHz, 30ms -> 480 samples.
        if frame_size not in [160, 320, 480]:
            # Adjust frame_size to a valid one, preferably closest to 30ms for 16kHz
            if self.config.sample_rate == 16000:
                frame_size = 480 # 30ms for 16kHz
            elif self.config.sample_rate == 8000:
                frame_size = 240 # 30ms for 8kHz
            else:
                logger.warning(f"Unsupported sample rate {self.config.sample_rate} for VAD. Skipping VAD.")
                return audio

        voiced_frames_list = []
        # Iterate through audio in chunks of `frame_size` samples
        for i in range(0, len(audio_16bit) - frame_size + 1, frame_size): # Ensure full frames
            frame = audio_16bit[i:i + frame_size]
            if len(frame) == frame_size:
                # VAD requires bytes
                is_speech = vad.is_speech(frame.tobytes(), self.config.sample_rate)
                if is_speech:
                    voiced_frames_list.extend(frame)
        
        if voiced_frames_list:
            return np.array(voiced_frames_list, dtype=np.float32) / 32767.0
        else:
            return audio  # Return original if no speech detected
    
    async def _detect_language(self, audio: np.ndarray) -> str:
        """Detect language from audio"""
        try:
            # Whisper can detect language automatically during transcription
            # Set language=None for auto-detection
            result: Dict[str, Any] = self.whisper_model.transcribe(audio, language=None, fp16=torch.cuda.is_available())
            detected_language = result.get("language", "en")
            
            # Map to our supported languages
            language_mapping = {
                "en": "en",
                "fr": "fr",
                "rw": "rw",
                "sw": "rw"  # Fallback Swahili to Kinyarwanda or a dedicated Swahili model if available
            }
            
            return language_mapping.get(detected_language, "en")
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}", exc_info=True)
            return "en"  # Default to English
    
    def _select_best_model(self, language: str) -> str:
        """Select the best ASR model for the detected language"""
        model_selection = {
            "en": "whisper_large", # Prefer Whisper for English
            "fr": "whisper_large", # Prefer Whisper for French
            "rw": "kinyarwanda_asr", # Prefer Kinyarwanda-specific Wav2Vec2 if available
            "auto": "whisper_large" # Default for auto-detection
        }
        
        # Check if the preferred model for the language is loaded, otherwise fallback
        selected_model = model_selection.get(language, "whisper_large")
        if selected_model not in self.models and selected_model != "whisper_large": # Whisper is loaded separately
            logger.warning(f"Preferred ASR model '{selected_model}' for language '{language}' not loaded. Falling back to Whisper.")
            return "whisper_large"
        
        return selected_model
    
    async def _transcribe_with_model(self, audio: np.ndarray, model_name: str, language: str) -> Dict[str, Any]:
        """Transcribe audio using specified model"""
        if model_name == "whisper_large":
            return await self._whisper_transcribe(audio, language)
        elif model_name in self.models: # Check if the model is actually loaded in self.models
            return await self._wav2vec2_transcribe(audio, model_name)
        else:
            logger.warning(f"Requested ASR model '{model_name}' not available. Falling back to Whisper.")
            return await self._whisper_transcribe(audio, language)
    
    async def _whisper_transcribe(self, audio: np.ndarray, language: str) -> Dict[str, Any]:
        """Transcribe using Whisper model"""
        try:
            # Whisper expects specific language codes, `None` for auto-detect or specific string
            whisper_lang_code = {"en": "en", "fr": "fr", "rw": "rw"}.get(language, None) # Use "rw" if Whisper supports it, else None
            
            result: Dict[str, Any] = self.whisper_model.transcribe(
                audio,
                language=whisper_lang_code,
                task="transcribe",
                fp16=torch.cuda.is_available()
            )
            
            return {
                "text": result.get("text", "").strip(),
                "confidence": float(self._calculate_whisper_confidence(result)),
                "language_confidence": 0.9,  # Whisper is generally confident
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}", exc_info=True)
            raise
    
    async def _wav2vec2_transcribe(self, audio: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Transcribe using Wav2Vec2 model"""
        try:
            processor = self.processors[model_name]
            model = self.models[model_name]
            
            # Process audio
            inputs = processor(audio, sampling_rate=self.config.sample_rate, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Get logits
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            # Calculate confidence (simplified - max softmax probability of predicted tokens)
            confidence = float(torch.softmax(logits, dim=-1).max().item())
            
            return {
                "text": transcription.strip(),
                "confidence": confidence,
                "language_confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed for {model_name}: {e}", exc_info=True)
            raise
    
    def _calculate_whisper_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from Whisper result"""
        if "segments" in result and result["segments"]:
            confidences = []
            for segment in result["segments"]:
                if "avg_logprob" in segment:
                    confidence = np.exp(segment["avg_logprob"]) # Convert log probability to confidence
                    confidences.append(confidence)
            
            if confidences:
                return float(np.mean(confidences))
        
        # Fallback confidence based on text length and quality
        text = result.get("text", "")
        if len(text.strip()) > 10:
            return 0.8
        elif len(text.strip()) > 5:
            return 0.6
        else:
            return 0.4
    
    async def _analyze_audio_characteristics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze various audio characteristics"""
        analysis = {}
        
        try:
            if self.config.emotion_detection:
                analysis["emotions"] = await self._analyze_emotions(audio)
            
            analysis["speech_rate"] = self._calculate_speech_rate(audio)
            analysis["quality"] = self._assess_audio_quality(audio)
            analysis["pauses"] = self._analyze_pauses(audio)
            
            if self.config.speaker_identification:
                analysis["speaker"] = await self._analyze_speaker_characteristics(audio)
            
        except Exception as e:
            logger.error(f"Error in audio characteristics analysis: {e}", exc_info=True)
        
        return analysis
    
    async def _analyze_emotions(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze emotional content of speech"""
        try:
            emotions = {
                "neutral": 0.4, "happy": 0.2, "sad": 0.1,
                "angry": 0.1, "fear": 0.1, "surprise": 0.1
            }
            
            mfccs = librosa.feature.mfcc(y=audio, sr=self.config.sample_rate, n_mfcc=13)
            energy = float(np.mean(librosa.feature.rms(y=audio)))
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.config.sample_rate)))
            
            if energy > 0.1 and spectral_centroid > 2000:
                emotions["happy"] += 0.3
                emotions["neutral"] -= 0.2
            elif energy < 0.05:
                emotions["sad"] += 0.2
                emotions["neutral"] -= 0.1
            
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            
            return emotions
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}", exc_info=True)
            return {"neutral": 1.0}
    
    def _calculate_speech_rate(self, audio: np.ndarray) -> float:
        """Calculate speech rate (words per minute)"""
        try:
            duration = len(audio) / self.config.sample_rate
            
            intervals = librosa.effects.split(audio, top_db=20)
            speech_duration = sum((end - start) / self.config.sample_rate for start, end in intervals)
            
            if speech_duration > 0:
                estimated_words = speech_duration * 2.5  # Rough syllable rate
                speech_rate = (estimated_words / speech_duration) * 60  # Words per minute
                return float(min(speech_rate, 300))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Speech rate calculation failed: {e}", exc_info=True)
            return 0.0
    
    def _assess_audio_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """Assess audio quality metrics"""
        try:
            quality = {}
            
            signal_power = float(np.mean(audio ** 2))
            noise_power = float(np.mean((audio - np.mean(audio)) ** 2) * 0.1)
            snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            quality["snr"] = float(max(0.0, min(1.0, snr / 30)))
            
            dynamic_range = float(np.max(audio) - np.min(audio))
            quality["dynamic_range"] = float(min(1.0, dynamic_range))
            
            fft = np.fft.fft(audio)
            spectral_energy = float(np.mean(np.abs(fft)))
            quality["spectral_quality"] = float(min(1.0, spectral_energy * 10))
            
            quality["overall"] = float(np.mean(list(quality.values())))
            
            return quality
            
        except Exception as e:
            logger.error(f"Audio quality assessment failed: {e}", exc_info=True)
            return {"overall": 0.5}
    
    def _analyze_pauses(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze pause patterns in speech"""
        try:
            intervals = librosa.effects.split(audio, top_db=20)
            
            if len(intervals) < 2:
                return {"pause_count": 0, "average_pause_duration": 0.0, "total_pause_time": 0.0, "pause_distribution": {}}
            
            pause_durations = []
            for i in range(len(intervals) - 1):
                pause_start = intervals[i][1]
                pause_end = intervals[i + 1][0]
                pause_duration = (pause_end - pause_start) / self.config.sample_rate
                if pause_duration > 0.1:  # Only count pauses longer than 100ms
                    pause_durations.append(pause_duration)
            
            return {
                "pause_count": len(pause_durations),
                "average_pause_duration": float(np.mean(pause_durations)) if pause_durations else 0.0,
                "total_pause_time": float(sum(pause_durations)),
                "pause_distribution": {
                    "short": sum(1 for p in pause_durations if p < 0.5),
                    "medium": sum(1 for p in pause_durations if 0.5 <= p < 1.0),
                    "long": sum(1 for p in pause_durations if p >= 1.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Pause analysis failed: {e}", exc_info=True)
            return {"pause_count": 0, "average_pause_duration": 0.0, "total_pause_time": 0.0, "pause_distribution": {}}
    
    async def _analyze_speaker_characteristics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze speaker characteristics"""
        try:
            characteristics = {}
            
            # Fundamental frequency (pitch) analysis
            # Ensure audio is mono and resampled for consistent F0 extraction
            f0 = librosa.yin(audio, fmin=50, fmax=400, sr=self.config.sample_rate)
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames
            
            if len(f0_clean) > 0:
                characteristics["average_pitch"] = float(np.mean(f0_clean))
                characteristics["pitch_range"] = float(np.max(f0_clean) - np.min(f0_clean))
                characteristics["pitch_variance"] = float(np.var(f0_clean))
            
            # Formant analysis (simplified - MFCCs are not directly formants but related to vocal tract shape)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.config.sample_rate, n_mfcc=13)
            # Taking mean of higher MFCCs might roughly correlate with vocal tract length, but it's a very rough estimate.
            characteristics["vocal_tract_length"] = float(np.mean(mfccs[1:4])) if mfccs.size > 0 else 0.0
            
            # Voice quality indicators
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.config.sample_rate)
            characteristics["brightness"] = float(np.mean(spectral_centroid)) if spectral_centroid.size > 0 else 0.0
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            characteristics["roughness"] = float(np.mean(zero_crossing_rate)) if zero_crossing_rate.size > 0 else 0.0
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Speaker characteristics analysis failed: {e}", exc_info=True)
            return {}

class AdvancedTTSEngine:
    """Advanced Text-to-Speech with emotion and voice cloning"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TTS model configurations
        self.model_configs = {
            "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2", # Coqui TTS model path
            "speecht5": "microsoft/speecht5_tts",
            "bark": "suno/bark", # Note: Bark might have specific installation requirements for audio backend
            "kinyarwanda_tts": "facebook/mms-tts-kin" # Example Kinyarwanda specific model (assuming compatibility with SpeechT5/AutoModel)
        }
        self.primary_tts: Any = None # Coqui TTS object
        self.speecht5_processor: Any = None
        self.speecht5_model: Any = None
        self.speecht5_vocoder: Any = None
        self.speaker_embeddings: Dict[str, torch.Tensor] = {} # Initialize speaker_embeddings
    
    async def initialize(self):
        """Initialize TTS models"""
        logger.info("Initializing Advanced TTS Engine...")
        
        try:
            # Load primary TTS model (XTTS v2 for multilingual support using Coqui TTS)
            logger.info(f"Loading primary TTS model (Coqui TTS XTTS v2): {self.config.voice_model} on {self.device}")
            self.primary_tts = TTS(self.config.voice_model).to(self.device)
            logger.info("Coqui TTS XTTS v2 model loaded.")
            
            # Load SpeechT5 for English (and potentially Kinyarwanda if compatible)
            logger.info("Loading SpeechT5 models...")
            self.speecht5_processor = SpeechT5Processor.from_pretrained(self.model_configs["speecht5"])
            self.speecht5_model = SpeechT5ForTextToSpeech.from_pretrained(self.model_configs["speecht5"]).to(self.device)
            self.speecht5_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
            logger.info("SpeechT5 models loaded.")

            # Load any custom/language-specific models if they are compatible with Hugging Face AutoModel
            if "kinyarwanda_tts" in self.model_configs:
                try:
                    # Assuming a Kinyarwanda model might be a fine-tuned SpeechT5 or similar
                    # For a custom Kinyarwanda model, this might need specialized loading
                    logger.info(f"Loading Kinyarwanda TTS model: {self.model_configs['kinyarwanda_tts']}")
                    # Example: if it's a HuggingFace compatible model
                    self.models["kinyarwanda_tts_processor"] = AutoProcessor.from_pretrained(self.model_configs["kinyarwanda_tts"])
                    self.models["kinyarwanda_tts_model"] = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_configs["kinyarwanda_tts"]).to(self.device)
                    logger.info("Kinyarwanda TTS model loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load Kinyarwanda TTS model: {e}")
            
            # Load speaker embeddings
            await self._load_speaker_embeddings()
            
            logger.info("TTS Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS Engine: {e}", exc_info=True)
            raise
    
    async def _load_speaker_embeddings(self):
        """Load pre-computed speaker embeddings or create defaults"""
        try:
            embeddings_path = Path("models/speaker_embeddings")
            if embeddings_path.exists():
                logger.info(f"Loading speaker embeddings from {embeddings_path}")
                for embedding_file in embeddings_path.glob("*.npy"):
                    speaker_name = embedding_file.stem
                    # Ensure embedding is loaded as float32 for consistency with PyTorch FloatTensor
                    embedding = np.load(embedding_file).astype(np.float32)
                    self.speaker_embeddings[speaker_name] = torch.tensor(embedding, dtype=torch.float32).to(self.device)
                logger.info(f"Loaded {len(self.speaker_embeddings)} speaker embeddings.")
            else:
                logger.warning(f"Speaker embeddings directory {embeddings_path} not found. Creating default embeddings.")
                # Create default embeddings (float32 for consistency)
                self.speaker_embeddings = {
                    "default": torch.randn(512, dtype=torch.float32).to(self.device),
                    "female": torch.randn(512, dtype=torch.float32).to(self.device),
                    "male": torch.randn(512, dtype=torch.float32).to(self.device)
                }
            
        except Exception as e:
            logger.error(f"Failed to load speaker embeddings: {e}", exc_info=True)
            self.speaker_embeddings = {"default": torch.randn(512, dtype=torch.float32).to(self.device)}
    
    async def synthesize_speech(self, text: str, language: str = "en", 
                              emotion_context: Optional[Dict[str, float]] = None,
                              speaker_profile: str = "default") -> bytes:
        """Synthesize speech with advanced features"""
        if self.primary_tts is None:
            raise RuntimeError("TTS Engine not initialized. Primary TTS model is not loaded.")

        try:
            # Select appropriate model and voice
            model_config = self._select_tts_model(language, emotion_context)
            
            # Preprocess text
            processed_text = await self._preprocess_text(text, language)
            
            # Generate speech
            audio: np.ndarray
            if model_config["model"] == "xtts_v2":
                audio = await self._xtts_synthesize(processed_text, language, emotion_context, speaker_profile)
            elif model_config["model"] == "speecht5":
                audio = await self._speecht5_synthesize(processed_text, speaker_profile)
            elif model_config["model"] == "kinyarwanda_tts":
                # Assuming kinyarwanda_tts uses a similar HF AutoModel for Seq2Seq
                audio = await self._kinyarwanda_tts_synthesize(processed_text)
            else:
                # Fallback to primary TTS (XTTS v2)
                audio = await self._xtts_synthesize(processed_text, language, emotion_context, speaker_profile)
            
            # Post-process audio
            enhanced_audio = await self._postprocess_audio(audio, emotion_context)
            
            # Convert to bytes
            audio_bytes = self._audio_to_bytes(enhanced_audio)
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}", exc_info=True)
            raise
    
    def _select_tts_model(self, language: str, emotion_context: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """Select the best TTS model for language and emotion"""
        # Model selection logic
        if language == "rw":
            if "kinyarwanda_tts_model" in self.models:
                return {"model": "kinyarwanda_tts", "voice": "kinyarwanda_speaker"}
            else:
                return {"model": "xtts_v2", "voice": "kinyarwanda_speaker"} # XTTS has multilingual
        elif language == "en":
            # For emotion-aware synthesis, Bark might be considered if loaded
            # The current setup only loads XTTS and SpeechT5 explicitly.
            if emotion_context and max(emotion_context.values()) > 0.7:
                # If Bark was loaded, one might use it here. For now, fallback.
                pass 
            return {"model": "speecht5", "voice": "default_speaker"} # Prefer SpeechT5 for English
        elif language == "fr":
            return {"model": "xtts_v2", "voice": "french_speaker"}
        else:
            return {"model": "xtts_v2", "voice": "multilingual_speaker"}
    
    async def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text for better TTS output"""
        processed_text = text.strip()
        
        if language == "rw":
            processed_text = self._normalize_kinyarwanda_text(processed_text)
        elif language == "en":
            processed_text = self._normalize_english_text(processed_text)
        elif language == "fr":
            processed_text = self._normalize_french_text(processed_text)
        
        processed_text = self._add_prosodic_markers(processed_text)
        
        return processed_text
    
    def _normalize_kinyarwanda_text(self, text: str) -> str:
        """Normalize Kinyarwanda text for TTS (Placeholder)"""
        return text
    
    def _normalize_english_text(self, text: str) -> str:
        """Normalize English text for TTS"""
        import re
        
        abbreviations = {
            "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss",
            "Prof.": "Professor", "etc.": "etcetera", "vs.": "versus"
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        text = re.sub(r'\b(\d+)\b', lambda m: self._number_to_words(int(m.group(1))), text)
        
        return text
    
    def _normalize_french_text(self, text: str) -> str:
        """Normalize French text for TTS (Placeholder)"""
        return text
    
    def _number_to_words(self, number: int) -> str:
        """Convert numbers to words (simplified placeholder)"""
        # For a full implementation, consider using num2words library
        if number == 0: return "zero"
        if number == 1: return "one"
        if number == 2: return "two"
        return str(number)
    
    def _add_prosodic_markers(self, text: str) -> str:
        """Add prosodic markers for better speech synthesis"""
        text = text.replace(".", ". <break time='0.5s'/>")
        text = text.replace(",", ", <break time='0.2s'/>")
        text = text.replace("?", "? <break time='0.7s'/>")
        text = text.replace("!", "! <break time='0.7s'/>")
        
        return text
    
    async def _xtts_synthesize(self, text: str, language: str, 
                             emotion_context: Optional[Dict[str, float]] = None,
                             speaker_profile: str = "default") -> np.ndarray:
        """Synthesize using XTTS v2 model (Coqui TTS)"""
        try:
            # XTTS requires a reference speaker_wav for cloning.
            # If not provided, it uses a default voice.
            # For simplicity, we're not using voice cloning here.
            # A real implementation would involve a default speaker or user-uploaded one.
            synthesis_kwargs: Dict[str, Any] = {
                "text": text,
                "language": language,
                # "speaker_wav": "/path/to/reference_audio.wav", # Uncomment for voice cloning
                "emotion": self._map_emotion_to_style(emotion_context) if emotion_context else "neutral"
            }
            
            # Generate audio
            audio = self.primary_tts.tts(**synthesis_kwargs)
            
            return np.array(audio)
            
        except Exception as e:
            logger.error(f"XTTS synthesis failed: {e}", exc_info=True)
            raise
    
    async def _speecht5_synthesize(self, text: str, speaker_profile: str = "default") -> np.ndarray:
        """Synthesize using SpeechT5 model"""
        if self.speecht5_processor is None or self.speecht5_model is None or self.speecht5_vocoder is None:
            raise RuntimeError("SpeechT5 models not initialized.")

        try:
            inputs = self.speecht5_processor(text=text, return_tensors="pt").to(self.device)
            
            speaker_embedding = self.speaker_embeddings.get(speaker_profile, self.speaker_embeddings["default"])
            speaker_embedding = speaker_embedding.unsqueeze(0) # Add batch dimension
            
            with torch.no_grad():
                # generate_speech directly returns a torch.Tensor
                speech: torch.Tensor = self.speecht5_model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embedding, # This is already a FloatTensor due to dtype=float32 earlier
                    vocoder=self.speecht5_vocoder
                )
            
            return speech.cpu().numpy()
            
        except Exception as e:
            logger.error(f"SpeechT5 synthesis failed: {e}", exc_info=True)
            raise

    async def _kinyarwanda_tts_synthesize(self, text: str) -> np.ndarray:
        """Synthesize using a custom Kinyarwanda TTS model (placeholder/example)"""
        processor_key = "kinyarwanda_tts_processor"
        model_key = "kinyarwanda_tts_model"
        
        if processor_key not in self.models or model_key not in self.models:
            logger.warning("Kinyarwanda TTS model or processor not loaded. Falling back to XTTS.")
            return await self._xtts_synthesize(text, "rw") # Fallback to XTTS multilingual
        
        try:
            processor = self.models[processor_key]
            model = self.models[model_key]

            inputs = processor(text=text, return_tensors="pt").to(self.device)
            
            # Depending on the model, speaker embeddings might be needed or it's implicitly handled.
            # For simplicity, using a default speaker embedding if required by the model.
            speaker_embedding = self.speaker_embeddings.get("default", torch.randn(512, dtype=torch.float32).to(self.device))
            speaker_embedding = speaker_embedding.unsqueeze(0)

            with torch.no_grad():
                # This assumes a structure compatible with generate_speech or similar method
                # This part is highly dependent on the specific Kinyarwanda model's API
                if hasattr(model, 'generate_speech'):
                     speech: torch.Tensor = model.generate_speech(
                        inputs["input_ids"], 
                        speaker_embedding, 
                        # vocoder=... # A separate vocoder might be needed depending on the model
                    )
                elif hasattr(model, 'generate'):
                    # Some models use 'generate'
                    generation_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs.get("attention_mask"),
                        # Add other required args like speaker_embeddings if needed
                        "speaker_embeddings": speaker_embedding if "speaker_embeddings" in model.forward.__code__.co_varnames else None,
                    }
                    if "vocoder" in model.forward.__code__.co_varnames:
                        generation_kwargs["vocoder"] = self.speecht5_vocoder # Use SpeechT5 vocoder as example
                    
                    speech_output = model.generate(**generation_kwargs)
                    speech = speech_output # Assume speech_output is the tensor directly
                else:
                    raise NotImplementedError("Kinyarwanda TTS model generation method not supported.")

            return speech.cpu().numpy()

        except Exception as e:
            logger.error(f"Kinyarwanda TTS synthesis failed: {e}", exc_info=True)
            raise
    
    def _map_emotion_to_style(self, emotion_context: Optional[Dict[str, float]]) -> str:
        """Map emotion analysis to TTS style"""
        if not emotion_context:
            return "neutral"
        
        dominant_emotion_item = max(emotion_context.items(), key=lambda x: x[1])
        dominant_emotion = dominant_emotion_item[0]
        
        emotion_mapping = {
            "happy": "cheerful", "sad": "sad", "angry": "angry",
            "fear": "fearful", "surprise": "surprised", "neutral": "neutral"
        }
        
        return emotion_mapping.get(dominant_emotion, "neutral")
    
    async def _postprocess_audio(self, audio: np.ndarray, emotion_context: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Post-process synthesized audio"""
        if emotion_context:
            audio = self._apply_emotional_processing(audio, emotion_context)
        
        audio = self._enhance_synthesized_audio(audio)
        audio = audio / np.max(np.abs(audio)) # Final normalization
        
        return audio
    
    def _apply_emotional_processing(self, audio: np.ndarray, emotion_context: Dict[str, float]) -> np.ndarray:
        """Apply emotion-based audio processing"""
        dominant_emotion_item = max(emotion_context.items(), key=lambda x: x[1])
        emotion, intensity = dominant_emotion_item
        
        # librosa.effects.pitch_shift requires sample rate
        sr_for_pitch_shift = 22050 # Assuming this is the output SR for TTS models
        
        if emotion == "happy" and intensity > 0.5:
            audio = librosa.effects.pitch_shift(audio, sr=sr_for_pitch_shift, n_steps=1)
        elif emotion == "sad" and intensity > 0.5:
            audio = librosa.effects.pitch_shift(audio, sr=sr_for_pitch_shift, n_steps=-1)
        elif emotion == "angry" and intensity > 0.5:
            audio = np.tanh(audio * 1.2) # Simple distortion
        
        return audio
    
    def _enhance_synthesized_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhance synthesized audio quality"""
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=22050, # Assuming this is the output SR for TTS models
            sample_width=audio.dtype.itemsize,
            channels=1
        )
        
        normalized = normalize(audio_segment)
        compressed = compress_dynamic_range(normalized, threshold=-15.0, ratio=2.0)
        
        enhanced_audio = np.array(compressed.get_array_of_samples(), dtype=np.float32)
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio))
        
        return enhanced_audio
    
    def _audio_to_bytes(self, audio: np.ndarray, format: str = "mp3") -> bytes:
        """Convert audio array to bytes"""
        audio_16bit = (audio * 32767).astype(np.int16) # Convert to 16-bit PCM
        
        audio_segment = AudioSegment(
            audio_16bit.tobytes(),
            frame_rate=22050, # Assuming this is the output SR for TTS models
            sample_width=2, # 16-bit PCM has 2 bytes per sample
            channels=1
        )
        
        buffer = io.BytesIO()
        audio_segment.export(buffer, format=format)
        return buffer.getvalue()

# --- Mocking external dependencies for self-contained testing ---
class MockCacheManager:
    """A mock cache manager for testing purposes."""
    def __init__(self):
        self.transcription_cache: Dict[str, Any] = {}
        self.tts_cache: Dict[str, bytes] = {}
        logger.info("MockCacheManager initialized.")

    async def get_transcription_cache(self, audio_hash: str) -> Optional[Dict[str, Any]]:
        logger.info(f"MockCacheManager: Getting transcription for {audio_hash}")
        return self.transcription_cache.get(audio_hash)

    async def cache_transcription(self, audio_hash: str, result: Dict[str, Any]):
        logger.info(f"MockCacheManager: Caching transcription for {audio_hash}")
        self.transcription_cache[audio_hash] = result

    async def get_tts_cache(self, tts_hash: str) -> Optional[bytes]:
        logger.info(f"MockCacheManager: Getting TTS audio for {tts_hash}")
        return self.tts_cache.get(tts_hash)

    async def cache_tts(self, tts_hash: str, audio_bytes: bytes):
        logger.info(f"MockCacheManager: Caching TTS audio for {tts_hash}")
        self.tts_cache[tts_hash] = audio_bytes

    async def close(self):
        logger.info("MockCacheManager closed.")
        self.transcription_cache.clear()
        self.tts_cache.clear()

    async def health_check(self) -> bool:
        return True

class MockSessionManager:
    """A mock session manager for testing purposes."""
    def __init__(self):
        logger.info("MockSessionManager initialized.")
    async def health_check(self) -> bool:
        return True

class MockModelOrchestrator:
    """A mock model orchestrator for testing purposes."""
    def __init__(self):
        logger.info("MockModelOrchestrator initialized.")
    async def health_check(self) -> bool:
        return True

class MockMonitoringEngine:
    """A mock monitoring engine for testing purposes."""
    def __init__(self):
        logger.info("MockMonitoringEngine initialized.")
    async def log_voice_metrics(self, metrics: Dict[str, Any]):
        logger.info(f"MockMonitoringEngine: Logged voice metrics: {metrics}")
    async def health_check(self) -> bool:
        return True

# --- Main Voice Processor Class (unchanged constructor, but now works with mocks) ---
class SophisticatedVoiceProcessor:
    """Main voice processor orchestrating ASR and TTS"""
    
    def __init__(self, cache_manager: MockCacheManager, session_manager: MockSessionManager, 
                 model_orchestrator: MockModelOrchestrator, monitoring_engine: MockMonitoringEngine):
        self.cache_manager = cache_manager
        self.session_manager = session_manager
        self.model_orchestrator = model_orchestrator
        self.monitoring_engine = monitoring_engine
        
        self.audio_config = AudioConfig()
        self.tts_config = TTSConfig()
        
        self.asr_engine = AdvancedASREngine(self.audio_config)
        self.tts_engine = AdvancedTTSEngine(self.tts_config)
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the sophisticated voice processor"""
        logger.info("Initializing Sophisticated Voice Processor...")
        
        try:
            # Initialize ASR engine
            await self.asr_engine.initialize()
            
            # Initialize TTS engine
            await self.tts_engine.initialize()
            
            self.is_initialized = True
            logger.info("Sophisticated Voice Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Sophisticated Voice Processor: {e}", exc_info=True)
            raise
    
    async def enhanced_transcribe_audio(self, audio_data: bytes, expected_language: str = "auto",
                                      session_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced audio transcription with comprehensive analysis"""
        if not self.is_initialized:
            raise RuntimeError("Voice Processor not initialized")
        
        try:
            # Check cache first
            audio_hash = hashlib.md5(audio_data).hexdigest()
            cached_result = await self.cache_manager.get_transcription_cache(audio_hash)
            if cached_result:
                return cached_result
            
            # Perform transcription
            voice_analysis = await self.asr_engine.transcribe_audio(audio_data, expected_language)
            
            # Prepare result
            result = {
                "text": voice_analysis.transcription,
                "confidence_score": voice_analysis.confidence_score,
                "detected_language": voice_analysis.detected_language,
                "language_confidence": voice_analysis.language_confidence,
                "emotion_analysis": voice_analysis.emotion_analysis,
                "speaker_characteristics": voice_analysis.speaker_characteristics,
                "audio_quality": voice_analysis.audio_quality,
                "speech_metrics": {
                    "speech_rate": voice_analysis.speech_rate,
                    "pause_analysis": voice_analysis.pause_analysis,
                    "pronunciation_assessment": voice_analysis.pronunciation_assessment
                }
            }
            
            # Cache result
            await self.cache_manager.cache_transcription(audio_hash, result)
            
            # Log metrics
            await self.monitoring_engine.log_voice_metrics({
                "session_id": session_id,
                "transcription_confidence": voice_analysis.confidence_score,
                "detected_language": voice_analysis.detected_language,
                "audio_duration": len(audio_data) / (self.audio_config.sample_rate * (self.audio_config.bit_depth / 8)),  # Corrected rough estimate
                "audio_quality": voice_analysis.audio_quality.get("overall", 0.5)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}", exc_info=True)
            raise
    
    async def enhanced_text_to_speech(self, text: str, language: str = "en",
                                    session_context: Optional[Dict[str, Any]] = None,
                                    emotion_context: Optional[Dict[str, float]] = None) -> bytes:
        """Enhanced text-to-speech with contextual adaptation"""
        if not self.is_initialized:
            raise RuntimeError("Voice Processor not initialized")
        
        try:
            # Check cache first
            tts_key = f"{text}_{language}_{str(emotion_context)}_{str(session_context)}" # Include session context in cache key
            tts_hash = hashlib.md5(tts_key.encode()).hexdigest()
            cached_audio = await self.cache_manager.get_tts_cache(tts_hash)
            if cached_audio:
                return cached_audio
            
            # Determine speaker profile from session context
            speaker_profile = "default"
            if session_context:
                speaker_profile = session_context.get("preferred_voice", "default")
            
            # Synthesize speech
            audio_bytes = await self.tts_engine.synthesize_speech(
                text=text,
                language=language,
                emotion_context=emotion_context,
                speaker_profile=speaker_profile
            )
            
            # Cache result
            await self.cache_manager.cache_tts(tts_hash, audio_bytes)
            
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Enhanced TTS failed: {e}", exc_info=True)
            raise
    
    async def health_check(self) -> bool:
        """Check health of voice processor components"""
        try:
            if not self.is_initialized:
                logger.warning("Voice Processor is not initialized.")
                return False
            
            # Check ASR engine
            if self.asr_engine.whisper_model is None:
                logger.error("ASR engine (Whisper model) is not loaded.")
                return False
            
            # Check TTS engine
            if self.tts_engine.primary_tts is None:
                logger.error("TTS engine (primary TTS model) is not loaded.")
                return False
            
            # Check mock dependencies health
            if not await self.cache_manager.health_check(): return False
            if not await self.session_manager.health_check(): return False
            if not await self.model_orchestrator.health_check(): return False
            if not await self.monitoring_engine.health_check(): return False

            logger.info("Voice Processor health check passed.")
            return True
            
        except Exception as e:
            logger.error(f"Voice processor health check failed: {e}", exc_info=True)
            return False

# --- Example Usage for Self-Contained Testing ---
async def main():
    """Example usage of the Sophisticated Voice Processor"""
    logging.basicConfig(level=logging.INFO) # Ensure logging is configured for main
    
    # Initialize mock dependencies
    mock_cache = MockCacheManager()
    mock_session = MockSessionManager()
    mock_model_orch = MockModelOrchestrator()
    mock_monitor = MockMonitoringEngine()

    # Initialize Voice Processor with mocks
    processor = SophisticatedVoiceProcessor(
        cache_manager=mock_cache,
        session_manager=mock_session,
        model_orchestrator=mock_model_orch,
        monitoring_engine=mock_monitor
    )

    try:
        # Initialize the processor (loads actual AI models)
        print("--- Initializing Voice Processor ---")
        await processor.initialize()
        
        # Health Check
        print("\n--- Running Health Check ---")
        is_healthy = await processor.health_check()
        print(f"Processor Healthy: {is_healthy}")
        if not is_healthy:
            print("Processor is not healthy, stopping tests.")
            return

        # --- Test ASR ---
        print("\n--- Testing ASR (Transcription) ---")
        # Generate dummy audio data (a simple sine wave for 2 seconds)
        duration = 2.0  # seconds
        frequency = 440  # Hz (A4 note)
        sample_rate = processor.audio_config.sample_rate # 16000 Hz
        num_samples = int(duration * sample_rate)
        t = np.linspace(0., duration, num_samples, endpoint=False)
        dummy_audio_np = 0.5 * np.sin(2. * np.pi * frequency * t) # amplitude 0.5

        # Convert numpy array to bytes (WAV format for robust processing)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, dummy_audio_np, sample_rate, format='WAV', subtype='PCM_16')
        dummy_audio_data = audio_buffer.getvalue()

        test_session_id = "test_session_123"
        print(f"Transcribing dummy audio (duration: {duration}s, SR: {sample_rate})...")
        transcription_result = await processor.enhanced_transcribe_audio(
            dummy_audio_data, expected_language="en", session_id=test_session_id
        )

        print("\nTranscription Result:")
        print(f"  Text: {transcription_result['text']}")
        print(f"  Confidence: {transcription_result['confidence']:.2f}")
        print(f"  Detected Language: {transcription_result['detected_language']}")
        print(f"  Audio Quality Overall: {transcription_result['audio_quality'].get('overall', 0.0):.2f}")
        
        # Test ASR from cache
        print("\n--- Testing ASR (from cache) ---")
        cached_transcription = await processor.enhanced_transcribe_audio(
            dummy_audio_data, expected_language="en", session_id=test_session_id
        )
        print(f"Cached Transcription Result Text: {cached_transcription['text']}")


        # --- Test TTS ---
        print("\n--- Testing TTS (Speech Synthesis) ---")
        text_to_synthesize = "Hello! This is a test message from Inyandiko Legal AI Assistant. Muraho!"
        print(f"Synthesizing speech for: '{text_to_synthesize}' in English...")
        
        # Example emotion context
        emotion_context_happy = {"happy": 0.8, "neutral": 0.2}
        
        synthesized_audio_en = await processor.enhanced_text_to_speech(
            text=text_to_synthesize, 
            language="en", 
            session_context={"user_id": "test_user"},
            emotion_context=emotion_context_happy
        )
        print(f"Synthesized English audio (length: {len(synthesized_audio_en)} bytes).")
        # Save to file to verify
        with open("output_en.mp3", "wb") as f:
            f.write(synthesized_audio_en)
        print("Saved English audio to output_en.mp3")

        text_to_synthesize_rw = "Murakoze cyane. Ntabwo ari ikibazo na gito." # Kinyarwanda: Thank you very much. It's not a problem at all.
        print(f"\nSynthesizing speech for: '{text_to_synthesize_rw}' in Kinyarwanda...")
        synthesized_audio_rw = await processor.enhanced_text_to_speech(
            text=text_to_synthesize_rw, 
            language="rw", 
            session_context={"user_id": "test_user"},
            emotion_context=None # Neutral emotion for Kinyarwanda test
        )
        print(f"Synthesized Kinyarwanda audio (length: {len(synthesized_audio_rw)} bytes).")
        with open("output_rw.mp3", "wb") as f:
            f.write(synthesized_audio_rw)
        print("Saved Kinyarwanda audio to output_rw.mp3")

        # Test TTS from cache
        print("\n--- Testing TTS (from cache) ---")
        cached_synthesized_audio = await processor.enhanced_text_to_speech(
            text=text_to_synthesize, 
            language="en", 
            session_context={"user_id": "test_user"},
            emotion_context=emotion_context_happy
        )
        print(f"Cached TTS audio length: {len(cached_synthesized_audio)} bytes (should be same as above).")
        
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Cleaning up ---")
        # No explicit cleanup needed for models in this simple test, but good practice for mocks
        await mock_cache.close()
        print("Testing complete.")

if __name__ == "__main__":
    asyncio.run(main())