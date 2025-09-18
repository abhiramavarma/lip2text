import pyaudio
import wave
import numpy as np
import threading
import time
from typing import Optional, Tuple


class AudioRecorder:
    """
    Audio recording utility that captures real-time audio from the system microphone.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024, 
                 format_type: int = pyaudio.paInt16):
        """
        Initialize PyAudio stream with configurable parameters.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000 for Whisper compatibility)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_size: Number of frames per buffer
            format_type: Audio format type (default: 16-bit PCM)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format_type = format_type
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Initialize recording state
        self.is_recording = False
        self.audio_frames = []
        self.recording_thread = None
    
    def start_recording(self) -> None:
        """
        Begin audio capture from default microphone.
        Stores audio data in memory buffer.
        """
        if self.is_recording:
            print("Already recording")
            return
        
        self.is_recording = True
        self.audio_frames = []
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print("Recording started")
    
    def _record(self) -> None:
        """
        Internal method to handle the recording process.
        """
        # Store actual sample rate for potential resampling
        self.actual_sample_rate = self.sample_rate
        stream = None
        
        try:
            # Try to open stream with requested parameters
            stream = self.p.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Error opening audio stream with requested parameters: {e}")
            
            # Try fallback to common sample rate (48kHz)
            try:
                print("Attempting fallback to 48kHz sample rate...")
                self.actual_sample_rate = 48000
                stream = self.p.open(
                    format=self.format_type,
                    channels=self.channels,
                    rate=self.actual_sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                print(f"Successfully opened audio stream with fallback rate: {self.actual_sample_rate}Hz")
            except Exception as e2:
                print(f"Error opening audio stream with fallback parameters: {e2}")
                print("Could not initialize audio recording. Check your microphone settings.")
                self.is_recording = False
                return
        
        if not stream:
            print("Failed to initialize audio stream")
            self.is_recording = False
            return
            
        try:
            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_frames.append(data)
        except Exception as e:
            print(f"Error during recording: {e}")
            self.is_recording = False
        finally:
            # Close the stream
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"Error closing audio stream: {e}")
    
    def stop_recording(self) -> Tuple[np.ndarray, int]:
        """
        Stop audio capture and return recorded audio data with sample rate.
        
        Returns:
            Tuple containing:
                - Numpy array containing the recorded audio data
                - Actual sample rate used for recording
        """
        if not self.is_recording:
            print("Not recording")
            return np.array([]), self.sample_rate
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()
        
        # Convert audio frames to numpy array
        audio_data = self._frames_to_array(self.audio_frames)
        
        print(f"Recording stopped. Captured {len(audio_data)/self.actual_sample_rate:.2f} seconds of audio")
        return audio_data, self.actual_sample_rate
    
    def _frames_to_array(self, frames) -> np.ndarray:
        """
        Convert audio frames to numpy array.
        
        Args:
            frames: List of audio frames
            
        Returns:
            Numpy array containing the audio data as float32 in range [-1, 1]
        """
        # Convert byte data to numpy array
        audio_data = b''.join(frames)
        
        # Convert to numpy array based on format
        if self.format_type == pyaudio.paInt16:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            # Normalize int16 to float32 in range [-1, 1]
            audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max
        elif self.format_type == pyaudio.paInt32:
            audio_np = np.frombuffer(audio_data, dtype=np.int32)
            # Normalize int32 to float32 in range [-1, 1]
            audio_np = audio_np.astype(np.float32) / np.iinfo(np.int32).max
        elif self.format_type == pyaudio.paInt8:
            audio_np = np.frombuffer(audio_data, dtype=np.int8)
            # Normalize int8 to float32 in range [-1, 1]
            audio_np = audio_np.astype(np.float32) / np.iinfo(np.int8).max
        elif self.format_type == pyaudio.paFloat32:
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            # Float32 is already in [-1, 1] range, just clip to be safe
            audio_np = np.clip(audio_np, -1.0, 1.0)
        else:
            # Default to int16 for unknown formats
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max
        
        # Handle multi-channel audio (convert to mono)
        if self.channels > 1:
            # Reshape to [frames, channels]
            audio_np = audio_np.reshape(-1, self.channels)
            # Average across channels to get mono
            audio_np = np.mean(audio_np, axis=1)
        
        return audio_np
    
    def save_audio(self, audio_data: np.ndarray, filename: str) -> None:
        """
        Save recorded audio to WAV file for processing.
        
        Args:
            audio_data: Numpy array containing audio data (1D for mono, 2D for multi-channel)
            filename: Output filename for the WAV file
        """
        # Determine number of channels based on audio_data shape
        if len(audio_data.shape) == 1:
            # 1D array - mono audio
            channels = 1
            # Use as-is
            processed_audio = audio_data
        else:
            # 2D array - multi-channel audio (frames, channels)
            channels = audio_data.shape[1]
            # Interleave by flattening row-major
            processed_audio = audio_data.reshape(-1)
        
        # Clip to [-1, 1] range
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        
        # Convert float32 to int16 for WAV file
        audio_int16 = (processed_audio * np.iinfo(np.int16).max).astype(np.int16)
        
        # Create WAV file with correct channel count
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        print(f"Audio saved to {filename}")
    
    def __del__(self):
        """
        Clean up PyAudio resources when the object is destroyed.
        """
        if hasattr(self, 'p'):
            self.p.terminate()