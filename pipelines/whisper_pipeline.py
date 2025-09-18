import os
import torch
import whisper
import numpy as np
from typing import Dict, Optional, Tuple, Union


class WhisperPipeline:
    """
    A dedicated pipeline for OpenAI Whisper speech-to-text transcription.
    Integrates with the existing audio processing infrastructure.
    """
    
    def __init__(self, model_size: str = "base", device: str = None):
        """
        Initialize Whisper model with configurable size.
        
        Args:
            model_size: Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_size = model_size
        print(f"Loading Whisper {model_size} model on {device}...")
        self.model = whisper.load_model(model_size).to(device)
        print(f"Whisper {model_size} model loaded successfully")
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """
        Load audio file and perform Whisper transcription.
        
        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language code to use for transcription
            
        Returns:
            Dictionary containing transcribed text and metadata
        """
        # Ensure the audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Set transcription options
        options = {}
        if language:
            options["language"] = language
        
        # Perform transcription
        result = self.model.transcribe(audio_path, **options)
        
        return result
    
    def transcribe_audio_from_array(self, audio_data: np.ndarray, sample_rate: int = 16000, 
                                   language: str = None) -> Dict:
        """
        Transcribe audio directly from a numpy array.
        
        IMPORTANT: This method expects audio data to be in 16kHz mono float32 format.
        If your audio doesn't match these requirements, call preprocess_audio() first.
        
        Args:
            audio_data: Audio data as numpy array (should be 16kHz mono float32)
            sample_rate: Sample rate of the audio data (should be 16000 Hz)
            language: Optional language code to use for transcription
            
        Returns:
            Dictionary containing transcribed text and metadata
        """
        # Ensure audio is properly formatted for Whisper
        if sample_rate != 16000 or audio_data.dtype != np.float32:
            audio_data = self.preprocess_audio(audio_data, sample_rate)
        
        # Set transcription options
        options = {}
        if language:
            options["language"] = language
        
        # Perform transcription directly from array
        result = self.model.transcribe(audio_data, **options)
        
        return result
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Preprocess audio data to match Whisper's expected format.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            Preprocessed audio data ready for Whisper
        """
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio to [-1, 1] range
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Use torchaudio for resampling
            import torchaudio
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            audio_data = audio_tensor.squeeze(0).numpy()
        
        return audio_data
    
    def get_available_models(self) -> list:
        """
        Returns a list of available Whisper model sizes.
        
        Returns:
            List of available model sizes
        """
        return ["tiny", "base", "small", "medium", "large"]