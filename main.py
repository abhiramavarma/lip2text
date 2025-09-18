import torch
import hydra
import cv2
import time
from pipelines.pipeline import InferencePipeline
import numpy as np
from datetime import datetime
from ollama import chat
from pydantic import BaseModel
import keyboard
from concurrent.futures import ThreadPoolExecutor
import os
from pipelines.audio_recorder import AudioRecorder
from pipelines.audio_utils import AudioUtils


# pydantic model for the chat output
class Lip2TextOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Lip2Text:
    def __init__(self):
        self.vsr_model = None
        
        # Mode management
        self.current_mode = "lip"  # "lip" or "voice"
        self.modes = ["lip", "voice"]

        # flag to toggle recording
        self.recording = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures = []

        # audio recorder - only initialize when needed
        self.audio_recorder = None
        
        # audio utils
        self.audio_utils = AudioUtils()

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

    def perform_inference(self, video_path):
        # perform inference on the video with the vsr model
        output = self.vsr_model(video_path)

        # write the raw output
        keyboard.write(output)

        # shift left to select the entire output
        cmd = ""
        for i in range(len(output)):
            cmd += 'shift+left, '
        cmd = cmd[:-2]
        keyboard.press_and_release(cmd)

        # perform inference on the raw output to get back a "correct" version
        response = chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'system',
                    'content': f"You are an assistant that helps make corrections to the output of a lipreading model. The text you will receive was transcribed using a video-to-text system that attempts to lipread the subject speaking in the video, so the text will likely be imperfect.\n\nIf something seems unusual, assume it was mistranscribed. Do your best to infer the words actually spoken, and make changes to the mistranscriptions in your response. Do not add more words or content, just change the ones that seem to be out of place (and, therefore, mistranscribed). Do not change even the wording of sentences, just individual words that look nonsensical in the context of all of the other words in the sentence.\n\nAlso, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'. The input text in all-caps, although your respose should be capitalized correctly and should NOT be in all-caps.\n\nReturn the corrected text in the format of 'list_of_changes' and 'corrected_text'."
                },
                {
                    'role': 'user',
                    'content': f"Transcription:\n\n{output}"
                }
            ],
            format=Lip2TextOutput.model_json_schema()
        )

        # get only the corrected text
        chat_output = Lip2TextOutput.model_validate_json(
            response.message.content)

        # if last character isn't a sentence ending (happens sometimes), add a period
        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # write the corrected text
        keyboard.write(chat_output.corrected_text + " ")

        # return the corrected text and the video path
        return {
            "output": chat_output.corrected_text,
            "video_path": video_path
        }

    def perform_audio_inference(self, audio_data_tuple):
        # Unpack the audio data and sample rate
        audio_data, sample_rate = audio_data_tuple
        
        # Transcribe audio using the audio_utils with the correct sample rate
        result = self.audio_utils.transcribe_audio(audio_data, save_temp=False, sample_rate=sample_rate)
        
        # Get the transcription text
        transcription = result.get('text', '')
        
        if not transcription:
            return {"output": "No speech detected"}
        
        # Write the raw output
        keyboard.write(transcription)
        
        # Select the entire output
        cmd = ""
        for i in range(len(transcription)):
            cmd += 'shift+left, '
        cmd = cmd[:-2]
        keyboard.press_and_release(cmd)
        
        # Perform inference on the raw output to get back a "correct" version
        response = chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'system',
                    'content': f"You are an assistant that helps make corrections to the output of a speech-to-text model. The text you will receive was transcribed using an audio-to-text system, so the text may contain errors.\n\nIf something seems unusual, assume it was mistranscribed. Do your best to infer the words actually spoken, and make changes to the mistranscriptions in your response. Do not add more words or content, just change the ones that seem to be out of place. Do not change the wording of sentences, just individual words that look nonsensical in the context.\n\nAlso, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'. The input text may be in all-caps, although your response should be capitalized correctly.\n\nReturn the corrected text in the format of 'list_of_changes' and 'corrected_text'."
                },
                {
                    'role': 'user',
                    'content': f"Transcription:\n\n{transcription}"
                }
            ],
            format=Lip2TextOutput.model_json_schema()
        )
        
        # Get only the corrected text
        chat_output = Lip2TextOutput.model_validate_json(
            response.message.content)
            
        # If last character isn't a sentence ending, add a period
        if chat_output.corrected_text and chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'
            
        # Write the corrected text
        keyboard.write(chat_output.corrected_text + " ")
        
        return {"output": chat_output.corrected_text}
        
    def start_webcam(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Create window and keep it always on top
        cv2.namedWindow('lip2text', cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty('lip2text', cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            # Some backends may not support TOPMOST; ignore if unavailable
            pass

        # set webcam resolution, and get frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_frame_time = time.time()

        # For tracking inference results
        output_path = ""
        out = None
        frame_count = 0

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # remove any remaining videos that were saved to disk
                for file in os.listdir():
                    if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                        os.remove(file)
                break

            current_time = time.time()

            # conditional ensures that the video is recorded at the correct frame rate
            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if ret:
                    # frame compression
                    encode_param = [
                        int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    compressed_frame = cv2.imdecode(
                        buffer, cv2.IMREAD_COLOR)
                    # Convert to grayscale for lip-reading (keep as single channel)
                    gray_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Validate frame dimensions match VideoWriter expectations
                    if gray_frame.shape[:2] != (frame_height, frame_width):
                        gray_frame = cv2.resize(gray_frame, (frame_width, frame_height))
                    
                    # Prepare display frame (flip first so overlay text isn't mirrored)
                    base_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    display_frame = cv2.flip(base_bgr, 1)

                    # Add mode text overlay with mode-specific colors (draw after flip)
                    mode_color = (0, 255, 0) if self.current_mode == "lip" else (0, 255, 255)
                    cv2.putText(display_frame, f"Mode: {self.current_mode.upper()}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                    # Add instructions (draw after flip so text is readable)
                    cv2.putText(display_frame, "Tab: Switch Mode | Alt: Record | Q: Quit",
                                (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1)

                    # Update recording indicator with mode-specific colors and behavior
                    if self.recording:
                        if self.current_mode == "lip":
                            # Black circle for lip mode recording
                            cv2.circle(display_frame, (frame_width - 20, 20), 10, (0, 0, 0), -1)
                            cv2.putText(display_frame, "REC VIDEO", (frame_width - 100, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        else:
                            # Red circle for voice mode recording
                            cv2.circle(display_frame, (frame_width - 20, 20), 10, (0, 0, 255), -1)
                            cv2.putText(display_frame, "REC AUDIO", (frame_width - 100, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    # Update last_frame_time before conditional blocks
                    last_frame_time = current_time
                    
                    # Guard for mode switching while recording video
                    if self.current_mode != "lip" and out is not None and frame_count > 0:
                        # Finalize video if we switched away from lip mode
                        last_video_path = output_path  # Capture path before releasing
                        last_frame_count = frame_count  # Capture frame count before reset
                        out.release()
                        out = None
                        
                        # Process or cleanup based on video length
                        if last_frame_count >= self.fps * 2:
                            self.futures.append(self.executor.submit(
                                self.perform_inference, last_video_path))
                            print(f"Video recording finalized due to mode switch - processing {last_video_path}")
                        else:
                            # Delete short video file safely
                            if os.path.exists(last_video_path):
                                os.remove(last_video_path)
                            print(f"Video recording finalized due to mode switch - deleted short segment")
                        
                        frame_count = 0
                    
                    # Group by mode first, then by recording state
                    if self.current_mode == "lip":
                        if self.recording:
                            if out is None:
                                output_path = self.output_prefix + \
                                    str(time.time_ns() // 1_000_000) + '.mp4'
                                out = cv2.VideoWriter(
                                    output_path,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    self.fps,
                                    (frame_width, frame_height),
                                    False  # isColor
                                )
                                
                                # Check if VideoWriter opened successfully
                                if not out.isOpened():
                                    print(f"Error: Failed to open VideoWriter for {output_path}")
                                    # Fallback codec option for future troubleshooting:
                                    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), self.fps, (frame_width, frame_height), False)
                                    out = None
                                    self.recording = False

                            out.write(gray_frame)  # Write grayscale frame, not BGR frame
                            frame_count += 1
                        # check if recording stopped AND frames were captured
                        elif not self.recording and frame_count > 0:
                            if out is not None:
                                out.release()

                            # only run inference if the video is at least 2 seconds long
                            if frame_count >= self.fps * 2:
                                self.futures.append(self.executor.submit(
                                    self.perform_inference, output_path))
                            else:
                                os.remove(output_path)

                            # Set out to None to avoid accidental reuse
                            out = None
                            frame_count = 0

                    # display the frame in the window (already flipped)
                    cv2.imshow('lip2text', display_frame)

            # ensures that videos and audio are handled in the order they were recorded
            for fut in list(self.futures):
                if fut.done():
                    result = fut.result()
                    if result:
                        # Handle both video and audio results
                        if "video_path" in result:
                            # Video result - delete video file
                            os.remove(result["video_path"])
                            print(f"Processed and cleaned up video: {result['output']}")
                        else:
                            # Audio result
                            print(f"Processed audio: {result['output']}")
                        self.futures.remove(fut)
                else:
                    break

        # release everything with graceful shutdown
        if self.current_mode == "voice" and self.audio_recorder and self.audio_recorder.is_recording:
            try:
                self.audio_recorder.stop_recording()
            except Exception as e:
                print(f"Error stopping audio on exit: {e}")

        # Finalize any in-progress video segment
        if self.current_mode == "lip" and frame_count > 0 and out is not None:
            try:
                out.release()
                print(f"Finalized in-progress video segment on exit")
            except Exception as e:
                print(f"Error finalizing video on exit: {e}")
        elif out is not None:
            try:
                out.release()
            except:
                pass

        cap.release()
        cv2.destroyAllWindows()

    def toggle_mode(self):
        """Toggle between lip and voice modes"""
        if self.recording:
            print("Cannot switch modes while recording. Press Alt to stop first.")
            return
            
        current_index = self.modes.index(self.current_mode)
        next_index = (current_index + 1) % len(self.modes)
        self.current_mode = self.modes[next_index]
        print(f"Switched to {self.current_mode.upper()} mode")
        
        # Initialize audio recorder only when switching to voice mode
        if self.current_mode == "voice" and self.audio_recorder is None:
            self.audio_recorder = AudioRecorder()
            
    def on_action(self, event):
        # Toggle mode when tab key is pressed
        if event.event_type == keyboard.KEY_DOWN and event.name == 'tab':
            self.toggle_mode()
        # Toggle recording when alt key is pressed (mode-specific behavior)
        elif event.event_type == keyboard.KEY_DOWN and event.name == 'alt':
            self.recording = not self.recording
            
            # Handle audio recording only in voice mode
            if self.current_mode == "voice":
                if self.recording:
                    if self.audio_recorder is None:
                        self.audio_recorder = AudioRecorder()
                    self.audio_recorder.start_recording()
                    print("Started audio recording")
                else:
                    # Stop recording and get audio data with sample rate
                    audio_data, sample_rate = self.audio_recorder.stop_recording()
                    
                    # Calculate duration in seconds using the actual sample rate
                    duration_sec = len(audio_data) / float(sample_rate)
                    
                    # Only process audio if it contains data and is at least 2 seconds long
                    if len(audio_data) > 0 and duration_sec >= 2.0:
                        # Submit audio for inference with both audio data and sample rate and track the future
                        self.futures.append(self.executor.submit(self.perform_audio_inference, (audio_data, sample_rate)))
                        print(f"Submitted audio for processing: {duration_sec:.2f}s")
                    else:
                        print(f"Audio too short ({duration_sec:.2f}s), discarding")


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    lip2text = Lip2Text()

    # hook to toggle recording
    keyboard.hook(lambda e: lip2text.on_action(e))

    # load the model
    lip2text.vsr_model = InferencePipeline(
        cfg.config_filename, device=torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available(
        ) and cfg.gpu_idx >= 0 else "cpu"), detector=cfg.detector, face_track=True)
    print("Model loaded successfully!")

    # start the webcam video capture
    lip2text.start_webcam()


if __name__ == '__main__':
    main()
