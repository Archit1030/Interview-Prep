import os
import io
from google.cloud import speech
from pydub import AudioSegment

class AudioEngine:
    def __init__(self):
        self.client = None
        
        try:
            # Try to find google_credentials.json in project root
            credentials_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'google_credentials.json')
            
            if os.path.exists(credentials_path):
                # Set environment variable to use this file
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
                print(f"✅ Found credentials at: {credentials_path}")
            
            # Initialize Google Cloud Speech-to-Text client
            self.client = speech.SpeechClient()
            print("✅ STT: Google Cloud Speech-to-Text Connected")
        except Exception as e:
            print(f"⚠️ STT Error: {e}")
            print("ℹ️ Falling back to basic STT. Place google_credentials.json in project root for better quality.")

    def calculate_volume(self, audio_chunk):
        """Calculates volume from pydub AudioSegment"""
        return float(round(audio_chunk.dBFS, 2))

    def process_audio(self, audio_bytes):
        """
        Takes raw bytes (WebM/WAV), converts to PCM WAV, then Transcribes using Google Cloud.
        Falls back to SpeechRecognition if Google Cloud is not available.
        """
        try:
            # 1. CONVERT AUDIO: Browser -> Clean WAV
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except Exception:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
            
            # Normalize audio
            audio_segment = audio_segment.normalize()
            
            # 2. GET VOLUME
            volume_db = audio_segment.dBFS
            volume_score = max(0, min(100, (volume_db + 60) * 2))

            # 3. TRANSCRIBE
            if self.client:
                # Google Cloud Speech-to-Text (High Quality)
                # Export to WAV format for Google Cloud
                wav_buffer = io.BytesIO()
                audio_segment.export(
                    wav_buffer, 
                    format="wav",
                    parameters=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz
                )
                wav_buffer.seek(0)
                audio_content = wav_buffer.read()

                # Configure recognition
                audio = speech.RecognitionAudio(content=audio_content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                    enable_automatic_punctuation=True,
                    model="latest_long",  # Best model for conversational speech
                )

                # Perform transcription
                response = self.client.recognize(config=config, audio=audio)
                
                # Extract text from response
                text = ""
                for result in response.results:
                    text += result.alternatives[0].transcript + " "
                text = text.strip()

            else:
                # Fallback to SpeechRecognition (Basic Quality)
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = 50   # lowered from 300
                recognizer.dynamic_energy_threshold = False
                
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                wav_buffer.seek(0)
                
                with sr.AudioFile(wav_buffer) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio_data = recognizer.record(source)
                
                text = recognizer.recognize_google(audio_data)

            # 4. ESTIMATE WPM (Words Per Minute)
            duration_sec = len(audio_segment) / 1000.0
            word_count = len(text.split())
            wpm = int((word_count / duration_sec) * 60) if duration_sec > 0 else 0

            return {
                "text": text,
                "volume": int(volume_score),
                "wpm": wpm,
                "error": None
            }

        except Exception as e:
            print(f"Audio Processing Error: {e}")
            return {"text": "", "volume": 0, "wpm": 0, "error": str(e)}