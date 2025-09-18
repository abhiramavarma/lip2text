## lip2text

lip2text is a local, privacy-first tool that can transcribe in two ways:

- Webcam lip-to-text: reads your lips in real time and types the words you silently mouth.
- Speech-to-text: captures your microphone audio and transcribes spoken words.

The app runs on your machine end-to-end. No cloud calls are made for lip reading or audio transcription.

### Features
- **Dual modes**: switch between lip-to-text and speech-to-text.
- **Always-on-top preview**: a small camera window stays above other apps.
- **Readable preview text**: the preview is mirrored correctly so overlays are not reversed.

### Requirements
- Python 3.12
- GPU optional (CUDA recommended for faster inference)
- The LRS3 visual model and subword language model files
- `ollama` with the `llama3.2` model (for optional punctuation/corrections)
- `uv` for running with the provided `requirements.txt`

### Setup
1. Clone and enter the repo:
   ```bash
   git clone <your-repo-url>
   cd lip2text
   ```
2. Download model artifacts:
   - Visual model: LRS3_V_WER19.1
   - Language model: lm_en_subword

   Place them like this:
   ```
   lip2text/
   ├── benchmarks/
       └── LRS3/
           ├── language_models/
           │   └── lm_en_subword/
           └── models/
               └── LRS3_V_WER19.1/
   ```
3. Install and run `ollama`, and pull `llama3.2`.
4. Install `uv`.

### Run
```bash
sudo uv run --with-requirements requirements.txt --python 3.12 main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
```

### How to use

By default, the app starts in lip-to-text mode.

- Switch modes: press `Tab` to toggle between lip-to-text (video) and speech-to-text (audio).
- Start/stop recording: press `Alt` (Windows/Linux) or `Option` (macOS).
- Quit: focus the preview window and press `q`.

#### Lip-to-Text (webcam)
1. Ensure your webcam is available.
2. When the preview appears, press `Alt/Option` to start capturing a short segment.
3. Mouth your words clearly; when you stop (or switch modes), the segment is processed.
4. The recognized text is typed into your active cursor location. A correction pass improves punctuation and small errors.

Tips:
- Face the camera with adequate lighting.
- Keep your face steady; avoid occluding your lips.

#### Speech-to-Text (microphone)
1. Press `Tab` to switch to voice mode.
2. Press `Alt/Option` to start recording. Speak normally.
3. Press `Alt/Option` again to stop. If the audio is at least ~2 seconds, it’s transcribed and typed at your cursor.

### Notes
- Local temporary video segments are cleaned up automatically after processing.
- If a recorded segment is shorter than ~2 seconds, it’s discarded.
- On systems without TOPMOST window support, the always-on-top hint may be ignored.

### Acknowledgments
- Visual speech recognition model from the Auto-AVSR project trained on LRS3.
