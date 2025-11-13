# RSTS: Real-Time Speech Translation System
This project implements a real-time speech recognition and translation system using FastAPI, WebSocket streaming, Whisper, and optional Google Cloud Translation. 
The system captures audio from the browser, sends it to the backend through WebSocket, performs ASR, and optionally translates or plays back TTS audio.

# Features
- Real-time microphone audio streaming through WebSocket.
- Speech recognition using Whisper (small model), running on macOS CPU or MPS.
- Optional translation through Google Cloud Translation API or local translation.
- Optional text-to-speech playback.
- Simple browser frontend with toggles for source text, translation, and TTS.

# Notes:  
The directory `translator_key/` and any API key JSON files are intentionally excluded using `.gitignore`.

# How to Run the Backend
Enter the backend folder and install dependencies:
pip install -r requirements.txt
Start the FastAPI server: python main.py
The backend will run at: http://localhost:8001
WebSocket endpoint: ws://localhost:8001/ws
How to Run the Frontend:  Open the file :frontend/index.html
 
# Requirements
Python 3.10 or newer
FastAPI
Whisper
Google Cloud Translation API (optional)
A modern browser that supports WebSocket and MediaRecorder

# API Keys
If you use Google Translation API, create a directory: translator_key/
Place your Google service account JSON file inside it.
Set the environment variable: export GOOGLE_APPLICATION_CREDENTIALS="translator_key/your_key.json"
This file will not be tracked by Git, as it is ignored in .gitignore.

# License
This project is intended for academic and research use.
