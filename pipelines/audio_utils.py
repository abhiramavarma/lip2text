import os
import tempfile
import numpy as np
import torch
import wave
import uuid
from typing import Dict, Optional, Tuple, Union

# Import local modules
from pipelines.whisper_pipeline import WhisperPipeline


def write_wav_mono_int16(filename: str, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
    """
    Write mono audio data to a WAV file.
    
    Args:
        filename: Output filename for the WAV file
        audio_data: Numpy array containing audio data (float32 in range [-1, 1])
        sample_rate: Audio sample rate in Hz
    """
    # Convert float32 to int16 for WAV file
    audio_int16 = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
    
    # Create WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Always mono for Whisper
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


class AudioUtils:
    """
    Audio processing utilities that bridge the gap between real-time audio recording and Whisper transcription.
    Integrates with the existing LLM correction workflow.
    """
    
    def __init__(self, whisper_model_size: str = "base", device: str = None, 
                 sample_rate: int = 16000, ollama_model: str = "llama3"):
        """
        Initialize audio utilities with configurable parameters.
        
        Args:
            whisper_model_size: Size of the Whisper model to use
            device: Device to run inference on
            sample_rate: Audio sample rate in Hz
            ollama_model: Ollama LLM model to use for correction
        """
        self.sample_rate = sample_rate
        self.ollama_model = ollama_model
        
        # Initialize Whisper pipeline
        self.whisper = WhisperPipeline(model_size=whisper_model_size, device=device)
        
        # Initialize temporary file storage
        self.temp_dir = tempfile.gettempdir()
        self.temp_files = []
    
    def convert_audio_for_whisper(self, audio_data: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """
        Convert audio data to format compatible with Whisper.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data, defaults to self.sample_rate if None
            
        Returns:
            Processed audio data ready for Whisper
        """
        # Use provided sample rate or fall back to default
        sr = sample_rate if sample_rate is not None else self.sample_rate
        
        # Preprocess audio using Whisper pipeline
        processed_audio = self.whisper.preprocess_audio(audio_data, sr)
        return processed_audio
    
    def save_temp_audio(self, audio_data: np.ndarray) -> str:
        """
        Save audio data to temporary file for processing.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Path to temporary audio file
        """
        # Create unique temporary WAV file using tempfile module
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=self.temp_dir).name
        
        # Use the helper function to write the WAV file
        write_wav_mono_int16(temp_file, audio_data, self.sample_rate)
        
        # Track temporary file for cleanup
        self.temp_files.append(temp_file)
        
        return temp_file
    
    def transcribe_audio(self, audio_data: np.ndarray, save_temp: bool = True, sample_rate: int = None) -> Dict:
        """
        Transcribe audio data using Whisper.
        
        Args:
            audio_data: Audio data as numpy array
            save_temp: Whether to save audio to temporary file
            sample_rate: Sample rate of the audio data, defaults to self.sample_rate if None
            
        Returns:
            Dictionary containing transcription results
        """
        if save_temp:
            # Save to temporary file and transcribe
            temp_file = self.save_temp_audio(audio_data)
            result = self.whisper.transcribe_audio(temp_file)
        else:
            # Transcribe directly from array
            processed_audio = self.convert_audio_for_whisper(audio_data, sample_rate)
            result = self.whisper.transcribe_audio_from_array(processed_audio, sample_rate=16000)
        
        return result
    
    def correct_transcription(self, transcription: str) -> str:
        """
        Send transcription to Ollama LLM for correction.
        
        Args:
            transcription: Raw transcription text from Whisper
            
        Returns:
            Corrected transcription text
        """
        try:
            import ollama
            
            # Prepare prompt for correction
            prompt = f"""
            Please correct any errors in this speech-to-text transcription.
            Fix grammar, punctuation, and any obvious word errors.
            Only output the corrected text, nothing else.
            
            Transcription: {transcription}
            """
            
            # Send to Ollama for correction
            response = ollama.chat(model=self.ollama_model, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            # Extract corrected text
            corrected_text = response['message']['content'].strip()
            return corrected_text
            
        except Exception as e:
            print(f"Error during LLM correction: {e}")
            # Return original transcription if correction fails
            return transcription
    
    def process_audio_to_text(self, audio_data: np.ndarray) -> str:
        """
        Complete pipeline from audio data to corrected text.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Corrected transcription text
        """
        # Transcribe audio
        result = self.transcribe_audio(audio_data)
        
        # Extract transcription text
        transcription = result.get('text', '')
        
        # Correct transcription using LLM
        corrected_text = self.correct_transcription(transcription)
        
        return corrected_text
    
    def cleanup_temp_audio(self) -> None:
        """
        Clean up temporary audio files.
        """
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up temporary file {temp_file}: {e}")
        
        # Clear the list of temporary files
        self.temp_files = []
    
    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        self.cleanup_temp_audio()


# Add missing import at the top
import time