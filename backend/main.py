import os
import sys
from groq import Groq
from huggingface_hub import InferenceClient
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
#import pygame
from PIL import Image
import os
import tempfile
import re
import warnings
import time
import base64
import io
import wave
import struct
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import re

# Try to import pydub, but don't fail if FFmpeg is missing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception as e:
    print(f"⚠️ pydub not available or FFmpeg missing: {e}")
    PYDUB_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

class MultilingualFarmerAgent:
    def __init__(self, groq_api_key=None, hf_token=None):
        # Initialize API clients (both 100% free)
        self.groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Groq for text (fast, free, unlimited)
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None
        
        # Hugging Face for images (free, no billing)
        self.hf_client = InferenceClient(token=self.hf_token) if self.hf_token else None
        
        self.recognizer = sr.Recognizer()
        
        print("✅ API Clients initialized (100% FREE)")
        if self.groq_client:
            print("  - Groq: Active (Text queries)")
        if self.hf_client:
            print("  - Hugging Face: Active (Image analysis)")
        # Initialize pygame for audio playback
        #pygame.mixer.init()

        # Updated language mapping to match Flutter app
        self.supported_languages = {
            'en': {'name': 'English', 'tts_code': 'en', 'display': 'English', 'sr_code': 'en-US'},
            'hi': {'name': 'हिन्दी', 'tts_code': 'hi', 'display': 'हिन्दी', 'sr_code': 'hi-IN'},
            'mr': {'name': 'मराठी', 'tts_code': 'mr', 'display': 'मराठी', 'sr_code': 'mr-IN'},
            'pa': {'name': 'ਪੰਜਾਬੀ', 'tts_code': 'pa', 'display': 'ਪੰਜਾਬੀ', 'sr_code': 'pa-IN'},
            'kn': {'name': 'ಕನ್ನಡ', 'tts_code': 'kn', 'display': 'ಕನ್ನಡ', 'sr_code': 'kn-IN'},
            'ta': {'name': 'தமிழ்', 'tts_code': 'ta', 'display': 'தமிழ்', 'sr_code': 'ta-IN'},
            'te': {'name': 'తెలుగు', 'tts_code': 'te', 'display': 'తెలుగు', 'sr_code': 'te-IN'},
            'ml': {'name': 'മലയാളം', 'tts_code': 'ml', 'display': 'മലയാളം', 'sr_code': 'ml-IN'},
            'gu': {'name': 'ગુજરાતી', 'tts_code': 'gu', 'display': 'ગુજરાતી', 'sr_code': 'gu-IN'},
            'bn': {'name': 'বাংলা', 'tts_code': 'bn', 'display': 'বাংলা', 'sr_code': 'bn-IN'},
            'or': {'name': 'ଓଡ଼ିଆ', 'tts_code': 'or', 'display': 'ଓଡ଼ିଆ', 'sr_code': 'or-IN'},
            'ur': {'name': 'اردو', 'tts_code': 'ur', 'display': 'اردو', 'sr_code': 'ur-PK'}
        }

        # Language name to code mapping for Flutter integration
        self.language_name_to_code = {
            'English': 'en',
            'हिन्दी': 'hi',
            'मराठी': 'mr',
            'ਪੰਜਾਬੀ': 'pa',
            'ಕನ್ನಡ': 'kn',
            'தமிழ்': 'ta',
            'తెలుగు': 'te',
            'മലയാളം': 'ml',
            'ગુજરાતી': 'gu',
            'বাংলা': 'bn',
            'ଓଡ଼ିଆ': 'or',
            'اردو': 'ur'
        }
    def _generate_content(self, prompt, image_path=None, max_retries=3):
        """
        Generate content with Groq (text) and Hugging Face (images).
        
        Key fix: image pipeline now does TWO BLIP passes (captioning + VQA)
        and stitches the results into a richer description that the LLM
        must use as the ONLY source of truth for crop identification.
        """
        
        # ── TEXT-ONLY PATH (fast, no image) ─────────────────────────────────
        if not image_path and self.groq_client:
            try:
                print("🔄 Using Groq API (text-only)...")
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Krishi Mitra, an expert agricultural advisor "
                                "for Indian farmers. Provide accurate, practical advice "
                                "in the requested language."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=2000
                )
                print("✅ Groq text response received")
                return response.choices[0].message.content
            
            except Exception as e:
                print(f"⚠️ Groq text failed: {e}")
        
        # ── IMAGE PATH ───────────────────────────────────────────────────────
        if image_path and self.hf_client:
            try:
                print("🔄 Analysing image with Hugging Face (enhanced pipeline)...")
        
                with open(image_path, "rb") as img_file:
                    image_bytes = img_file.read()
        
                # ── Pass 1: General caption ──────────────────────────────
                try:
                    caption_result = self.hf_client.image_to_text(
                        image_bytes,
                        model="Salesforce/blip-image-captioning-large"
                    )
                    # BLIP returns a string directly
                    general_caption = (
                        caption_result
                        if isinstance(caption_result, str)
                        else getattr(caption_result, "generated_text", str(caption_result))
                    )
                    print(f"📷 BLIP caption: {general_caption}")
                except Exception as blip_err:
                    print(f"⚠️ BLIP caption failed: {blip_err}")
                    general_caption = ""
        
                # ── Pass 2: VQA – ask specific plant-identity questions ───
                vqa_answers = {}
                vqa_questions = [
                    ("What type of plant or tree is shown in this image?",        "plant_type"),
                    ("What is the color and shape of the leaves?",                 "leaf_desc"),
                    ("Are there any fruits, flowers, or pods visible?",            "fruit_flower"),
                    ("Does the plant show any disease, spots, or yellowing?",      "health"),
                    ("What is the overall size — small herb, shrub, or tall tree?","size"),
                ]
        
                if self.hf_client:
                    for question, key in vqa_questions:
                        try:
                            answer = self.hf_client.visual_question_answering(
                                image=image_bytes,
                                question=question,
                                model="dandelin/vilt-b32-finetuned-vqa"
                            )
                            # answer is a list of dicts [{"answer": ..., "score": ...}]
                            if isinstance(answer, list) and answer:
                                best = max(answer, key=lambda x: x.get("score", 0))
                                vqa_answers[key] = best.get("answer", "unknown")
                            elif isinstance(answer, dict):
                                vqa_answers[key] = answer.get("answer", "unknown")
                            else:
                                vqa_answers[key] = str(answer)
                            print(f"  VQA [{key}]: {vqa_answers[key]}")
                        except Exception as vqa_err:
                            print(f"  ⚠️ VQA [{key}] failed: {vqa_err}")
                            vqa_answers[key] = "unable to determine"
        
                # ── Build a rich structured image description ─────────────
                image_description = f"""
STRICT IMAGE EVIDENCE (derived directly from the uploaded photograph):
  General scene  : {general_caption if general_caption else 'Not available'}
  Plant / tree   : {vqa_answers.get('plant_type', 'Not determined')}
  Leaf details   : {vqa_answers.get('leaf_desc', 'Not determined')}
  Fruit / flowers: {vqa_answers.get('fruit_flower', 'Not determined')}
  Plant health   : {vqa_answers.get('health', 'Not determined')}
  Plant size     : {vqa_answers.get('size', 'Not determined')}
""".strip()
        
                print(f"\n📋 Compiled image description:\n{image_description}\n")
        
                # ── Now call Groq with a prompt that ENFORCES the evidence ─
                if self.groq_client:
                    combined_prompt = f"""
{image_description}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTION — READ BEFORE RESPONDING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST identify the crop or plant using ONLY the evidence listed above.
Do NOT assume, guess, or name any crop (e.g. soybean, wheat, rice) that
is NOT explicitly supported by the image evidence above.
If the evidence says "mango tree", "large leafy tree", "tree with oval
green leaves and yellow fruit" — then it IS a mango tree.
If the plant type is unclear, say so honestly and advise the farmer to
consult a local expert rather than naming a random crop.

━━━━━━━━━━━━━━━━━
FARMER'S QUESTION:
━━━━━━━━━━━━━━━━━
{prompt}

Now answer the farmer's question based on the image evidence above.
"""
                    response = self.groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are Krishi Mitra, an expert agricultural advisor. "
                                    "You only identify crops from DIRECT IMAGE EVIDENCE provided "
                                    "to you. You never hallucinate or assume crop types. "
                                    "If image evidence is ambiguous, you say so clearly."
                                )
                            },
                            {"role": "user", "content": combined_prompt}
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.5,
                        max_tokens=2000
                    )
                    print("✅ Combined image+text response received")
                    return response.choices[0].message.content
        
            except Exception as e:
                print(f"⚠️ Image pipeline failed: {e}")
        
        # ── FALLBACK: text-only ──────────────────────────────────────────────
        if self.groq_client:
            try:
                print("🔄 Fallback to Groq text-only...")
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are Krishi Mitra, an agricultural expert."},
                        {"role": "user",   "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Fallback failed: {e}")
        
        return "सेवा अस्थायी रूप से अनुपलब्ध है। कृपया बाद में प्रयास करें। (Service temporarily unavailable)"
    
    def get_language_code_from_name(self, language_name):
        """Convert language name to language code"""
        return self.language_name_to_code.get(language_name, 'en')

    def get_language_name(self, language_code):
        """Get language name from code"""
        return self.supported_languages.get(language_code, {}).get('name', 'English')

    def get_supported_languages(self):
        """Return supported languages for the dropdown"""
        return [
            {'code': code, 'name': data['name'], 'display': data['display']}
            for code, data in self.supported_languages.items()
        ]

    def save_base64_audio_to_file(self, base64_data):
        """Enhanced audio file saving with better error handling"""
        try:
            print(f"📥 Received base64 audio data (length: {len(base64_data)} chars)")

            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                header, base64_data = base64_data.split(',', 1)
                print(f"📋 Detected data URL header: {header}")

            # Decode base64 data
            try:
                audio_bytes = base64.b64decode(base64_data)
                print(f"✅ Successfully decoded base64 data: {len(audio_bytes)} bytes")
            except Exception as decode_error:
                print(f"❌ Base64 decode error: {decode_error}")
                return None

            # Create a unique temporary file with proper suffix
            timestamp = int(time.time() * 1000)
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"audio_{timestamp}.wav")

            print(f"📁 Temp directory: {temp_dir}")
            print(f"📝 Creating temp file: {temp_path}")

            # Write bytes to file
            try:
                with open(temp_path, 'wb') as temp_file:
                    temp_file.write(audio_bytes)

                # Verify file was created and has content
                if os.path.exists(temp_path):
                    file_size = os.path.getsize(temp_path)
                    print(f"✅ Audio file created successfully: {temp_path}")
                    print(f"📊 File size: {file_size} bytes")

                    if file_size == 0:
                        print("⚠️ Warning: Audio file is empty")
                        os.unlink(temp_path)
                        return None

                    return temp_path
                else:
                    print("❌ File was not created")
                    return None

            except Exception as write_error:
                print(f"❌ File write error: {write_error}")
                return None

        except Exception as e:
            print(f"❌ Error in save_base64_audio_to_file: {e}")
            return None

    def is_valid_wave_file(self, file_path):
        """Check if the file is a valid WAV file"""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                # Get basic info
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()

                print(f"📊 WAV file info: {frames} frames, {sample_rate}Hz, {channels} channels, {sample_width} bytes/sample")

                # Check if it has actual audio data
                if frames > 0:
                    return True
                else:
                    print("⚠️ WAV file has no audio frames")
                    return False

        except Exception as e:
            print(f"❌ Invalid WAV file: {e}")
            return False

    def convert_to_wav_basic(self, input_path):
        """Basic conversion without FFmpeg - works with simple formats"""
        try:
            print(f"🔄 Attempting basic WAV conversion for: {input_path}")

            # First, check if it's already a valid WAV file
            if self.is_valid_wave_file(input_path):
                print("✅ File is already a valid WAV file")
                return input_path

            # Try to read as raw audio data and create a proper WAV file
            with open(input_path, 'rb') as f:
                audio_data = f.read()

            # Check if this looks like it might be raw PCM data
            if len(audio_data) > 44:  # Must be larger than WAV header
                # Try to create a WAV file assuming 16kHz, mono, 16-bit
                output_path = input_path.replace('.wav', '_fixed.wav')

                # Create WAV header for 16kHz, mono, 16-bit PCM
                sample_rate = 16000
                channels = 1
                bits_per_sample = 16

                # Skip potential existing headers and try to find PCM data
                # Look for patterns that suggest this is audio data
                potential_data = audio_data[44:] if len(audio_data) > 44 else audio_data

                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(bits_per_sample // 8)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(potential_data)

                # Test if the created file is valid
                if self.is_valid_wave_file(output_path):
                    print(f"✅ Successfully created WAV file: {output_path}")
                    return output_path
                else:
                    print("❌ Created WAV file is not valid")
                    if os.path.exists(output_path):
                        os.unlink(output_path)

            print("❌ Could not convert to valid WAV format")
            return None

        except Exception as e:
            print(f"❌ Basic WAV conversion error: {e}")
            return None

    def convert_audio_format(self, input_path):
        """Convert audio with fallback methods"""
        try:
            print(f"🔄 Converting audio format from {input_path}")

            # Method 1: Check if it's already a valid WAV
            if self.is_valid_wave_file(input_path):
                print("✅ File is already a valid WAV file")
                return input_path

            # Method 2: Try pydub if available
            if PYDUB_AVAILABLE:
                try:
                    print("🔄 Trying pydub conversion...")
                    audio = AudioSegment.from_file(input_path)

                    # Convert to optimal format for speech recognition
                    audio = audio.set_frame_rate(16000)
                    audio = audio.set_channels(1)
                    audio = audio.set_sample_width(2)

                    # Create output path
                    output_path = input_path.replace('.wav', '_pydub.wav')
                    audio.export(output_path, format="wav")

                    if self.is_valid_wave_file(output_path):
                        print(f"✅ pydub conversion successful: {output_path}")
                        return output_path
                    else:
                        print("❌ pydub conversion produced invalid file")
                        if os.path.exists(output_path):
                            os.unlink(output_path)

                except Exception as pydub_error:
                    print(f"❌ pydub conversion failed: {pydub_error}")

            # Method 3: Try basic conversion
            converted_path = self.convert_to_wav_basic(input_path)
            if converted_path:
                return converted_path

            # Method 4: Try to use the original file directly
            print("⚠️ Using original file without conversion")
            return input_path

        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return input_path

    def speech_to_text_with_language(self, audio_file_path, language_code):
        """Enhanced speech recognition with better error handling"""
        print(f"🎤 Processing audio file: {audio_file_path} for language: {language_code}")

        # Check if file exists
        if not os.path.exists(audio_file_path):
            error_msg = f"Audio file not found: {audio_file_path}"
            print(f"❌ {error_msg}")
            return error_msg

        # Get file size
        try:
            file_size = os.path.getsize(audio_file_path)
            print(f"📊 Original file size: {file_size} bytes")

            if file_size == 0:
                return "Audio file is empty"

        except Exception as e:
            print(f"❌ Error checking file size: {e}")
            return f"Error accessing audio file: {e}"

        # Get the speech recognition code for the selected language
        lang_data = self.supported_languages.get(language_code, self.supported_languages['en'])
        sr_code = lang_data['sr_code']
        lang_name = lang_data['name']

        print(f"🌐 Using language: {lang_name} ({sr_code})")

        # Convert audio to compatible format
        converted_path = self.convert_audio_format(audio_file_path)
        if not converted_path:
            return "Could not process audio file format"

        try:
            # Try multiple approaches for loading the audio file
            audio_data = None

            # Approach 1: Direct WAV file loading
            if self.is_valid_wave_file(converted_path):
                try:
                    with sr.AudioFile(converted_path) as source:
                        print("✅ Audio file loaded as WAV")
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio_data = self.recognizer.record(source)
                except Exception as wav_error:
                    print(f"❌ WAV loading failed: {wav_error}")

            # Approach 2: Try loading as different formats
            if audio_data is None:
                for file_ext in ['.wav', '.flac', '.aiff']:
                    try:
                        temp_renamed = converted_path + file_ext
                        if converted_path != temp_renamed:
                            # Copy to temp file with different extension
                            import shutil
                            shutil.copy2(converted_path, temp_renamed)

                            with sr.AudioFile(temp_renamed) as source:
                                print(f"✅ Audio loaded as {file_ext}")
                                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                                audio_data = self.recognizer.record(source)
                                break

                    except Exception as ext_error:
                        print(f"❌ Failed to load as {file_ext}: {ext_error}")
                        continue
                    finally:
                        # Clean up temp renamed file
                        if 'temp_renamed' in locals() and os.path.exists(temp_renamed):
                            try:
                                os.unlink(temp_renamed)
                            except:
                                pass

            if audio_data is None:
                return "Could not load audio file for speech recognition"

            print("📼 Audio data ready for recognition")

            # Try recognition with the specified language
            try:
                print(f"🔍 Attempting recognition for {lang_name}...")
                text = self.recognizer.recognize_google(
                    audio_data,
                    language=sr_code,
                    show_all=False
                )

                if text and text.strip():
                    print(f"✅ Recognition successful: '{text}'")
                    return text.strip()
                else:
                    return f"No speech detected in the {lang_name} audio"

            except sr.UnknownValueError:
                print(f"⚠️ Could not understand {lang_name} audio, trying fallback...")

                # Try with English as fallback
                try:
                    text = self.recognizer.recognize_google(audio_data, language='en-US')
                    if text and text.strip():
                        print(f"✅ Fallback recognition (English): '{text}'")
                        # Translate to target language if needed
                        if language_code != 'en':
                            try:
                               translated=GoogleTranslator(source='auto', target=language_code).translate(text)
                               return f"{text} (Translated: {translated})"
                            except:
                                return text
                        return text.strip()
                except sr.UnknownValueError:
                    return f"Could not understand the audio in any language"

            except sr.RequestError as e:
                print(f"❌ Google Speech Recognition error: {e}")
                return f"Speech recognition service error: {e}"

        except Exception as e:
            error_msg = f"Error processing audio file: {e}"
            print(f"❌ {error_msg}")
            return error_msg

        finally:
            # Clean up converted file if it's different from original
            if converted_path != audio_file_path and os.path.exists(converted_path):
                try:
                    os.unlink(converted_path)
                    print("🗑️ Cleaned up converted audio file")
                except:
                    pass
    # Add this enhanced method to your MultilingualFarmerAgent class

    def get_contextual_data(self, query_text, location_info=None):
        """Get relevant contextual data based on farmer's query"""
        query_lower = query_text.lower()
        contextual_data = {}

        # Default location if not provided
        if not location_info:
            location_info = {
                'city': 'Mumbai',
                'state': 'Maharashtra',
                'district': 'Pune'
            }

        # Check if query is weather-related
        weather_keywords = ['weather', 'rain', 'temperature', 'humidity', 'wind', 'climate',
                           'monsoon', 'drought', 'cold', 'hot', 'sunny', 'cloudy', 'storm',
                           'मौसम', 'बारिश', 'तापमान', 'हवा', 'सूखा', 'ठंड', 'गर्मी']

        needs_weather = any(keyword in query_lower for keyword in weather_keywords)

        # Check if query is market/price-related
        market_keywords = ['price', 'market', 'sell', 'buy', 'cost', 'rate', 'value', 'money',
                          'किमत', 'बाजार', 'बेचना', 'खरीदना', 'दाम', 'रुपया', 'पैसा']

        needs_market = any(keyword in query_lower for keyword in market_keywords)

        # Always get weather for crop-related queries (it's almost always relevant)
        crop_keywords = ['crop', 'plant', 'grow', 'harvest', 'seed', 'irrigation', 'fertilizer',
                        'फसल', 'पौधा', 'उगाना', 'बीज', 'सिंचाई', 'खाद']

        if any(keyword in query_lower for keyword in crop_keywords):
            needs_weather = True

        try:
            # Get weather data if needed
            if needs_weather:
                weather_scraper = WeatherScraper()
                weather_data = weather_scraper.get_weather_data(
                    location_info['city'],
                    location_info['state']
                )
                contextual_data['weather'] = weather_data
                print(f"✅ Retrieved weather data for context")

            # Get market data if needed
            if needs_market:
                market_scraper = MarketPriceScraper()
                market_data = market_scraper.get_commodity_prices(
                    location_info['state'],
                    location_info['district']
                )
                contextual_data['market'] = market_data
                print(f"✅ Retrieved market data for context")

        except Exception as e:
            print(f"⚠️ Error getting contextual data: {e}")

        return contextual_data

    # Update the existing process_farmer_query method
    def process_farmer_query_enhanced(self, image_path=None, audio_path=None, language_code="hi", location_info=None):
        """Enhanced main function with contextual data integration"""
        try:
            query_text = ""

            if audio_path:
                print("\n" + "="*60)
                print(f"🌾 KRISHI MITRA - Enhanced Processing in {self.get_language_name(language_code)}")
                print("="*60)

                query_text = self.speech_to_text_with_language(audio_path, language_code)

                if "Error" in query_text or "not found" in query_text or "Could not understand" in query_text:
                    return {"error": query_text, "language": language_code, "success": False}

                print(f"\n📝 Farmer's Query: {query_text}")
                print(f"🌐 Language: {self.get_language_name(language_code)} ({language_code})")

            if image_path:
                print(f"\n🌱 Analyzing crop image with real-time data...")

                analysis = self.analyze_crop_image_with_context(
                    image_path, query_text, language_code, location_info
                )

                return {
                    "query": query_text,
                    "language": language_code,
                    "language_name": self.get_language_name(language_code),
                    "analysis": analysis,
                    "success": True
                }
            elif query_text:
                print(f"\n💬 Processing text query with real-time context...")

                analysis = self.process_text_query_with_context(
                    query_text, language_code, location_info
                )

                return {
                    "query": query_text,
                    "language": language_code,
                    "language_name": self.get_language_name(language_code),
                    "analysis": analysis,
                    "success": True
                }
            else:
                return {"error": "No valid input provided", "success": False}

        except Exception as e:
            print(f"Error in enhanced process_farmer_query: {e}")
            return {"error": f"Processing error: {e}", "success": False}

    def process_text_query_with_context(self, text, target_language="hi", location_info=None):
            """Enhanced text processing with real-time data integration (FINAL FIXES APPLIED)"""
            try:
                # Get contextual data first
                contextual_data = self.get_contextual_data(text, location_info)

                lang_data = self.supported_languages.get(target_language, self.supported_languages['hi'])
                lang_name = lang_data['name']

                # Build context string for prompt
                context_string = ""

                # --- 1. ADD WEATHER CONTEXT (Independent Check) ---
                if 'weather' in contextual_data and contextual_data['weather'].get('success'):
                    weather = contextual_data['weather']
                    current = weather.get('current', {})
                    advice = weather.get('agricultural_advice', [])

                    context_string += f"""
                CURRENT WEATHER DATA for {weather.get('location', 'your area')}:
                - Temperature: {current.get('temperature', 'N/A')}°C
                - Humidity: {current.get('humidity', 'N/A')}%
                - Condition: {current.get('description', 'N/A')}
                - Wind Speed: {current.get('wind_speed', 'N/A')} km/h

                Weather-based Agricultural Advice:
                {chr(10).join(f'• {tip}' for tip in advice[:3])}
                """
                # --- 2. ADD MARKET CONTEXT (Independent Check - FINAL ROBUST LOGIC) ---
                market_data_added = False

                if 'market' in contextual_data and contextual_data['market'].get('success'):
                    market = contextual_data['market']
                    prices = market.get('data', [])[:5]  # Top 5 commodities

                    market_text_buffer = ""

                    # Start the market data section header
                    market_text_buffer += f"""
    CURRENT MARKET PRICES for {market.get('location', 'your area')}:
    """
                    for item in prices:
                        if isinstance(item, dict):
                            # Find the appropriate price key
                            if 'modal_price' in item:
                                price_key = 'modal_price'
                                unit = item.get('unit', 'quintal')
                                change = 'N/A'
                            elif 'price_per_kg' in item:
                                price_key = 'price_per_kg'
                                unit = item.get('unit', 'kg')
                                change = item.get('change', 'stable')
                            else:
                                continue # Skip items without a recognizable price

                            # Check if the name exists
                            if 'name' in item:
                                # Format the price line
                                item_name = item['name']
                                price_value = item[price_key]

                                # Highlight the tomato price if requested
                                is_target_commodity = 'tomato' in text.lower() and 'tomato' in item_name.lower()

                                if is_target_commodity:
                                    market_text_buffer += f"• **TOMATO (FOCUSED):** ₹{price_value}/{unit} ({change})\n"
                                else:
                                    market_text_buffer += f"• {item_name}: ₹{price_value}/{unit} ({change})\n"

                                market_data_added = True

                    # Only append the buffered text if we actually found commodity data
                    if market_data_added:
                        context_string += market_text_buffer
                    else:
                        # Provide a fallback status if the API succeeded but data was empty
                        context_string += f"""
                CURRENT MARKET PRICES for {market.get('location', 'your area')}:
                • Price information was retrieved but no specific commodity data was found.
                """
                # --- END: MARKET CONTEXT ---

                # Add this print statement for debugging
                print("\n--- CONTEXT SENT TO GEMINI ---")

                # Language-specific prompts (remaining unchanged)
                language_prompts = {
                    'pa': "ਕਿਰਪਾ ਕਰਕੇ ਸਿਰਫ਼ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ। ਕੋਈ ਅੰਗਰੇਜ਼ੀ ਸ਼ਬਦ ਨਹੀਂ।",
                    'gu': "કૃપા કરીને ફક્ત ગુજરાતીમાં જ જવાબ આપો। કોઈ અંગ્રેજી શબ્દો નહીં.",
                    'mr': "कृपया फक्त मराठीत उत्तर द्या. इंग्रजी শব্দ नाहीत.",
                    'kn': "ದಯವಿಟ್ಟು ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಿ। ಯಾವುದೇ ಇಂಗ್ಲಿಷ್ ಪದಗಳಿಲ್ಲ.",
                    'ta': "தயவு செய்து தமிழில் மட்டுமே பதிலளிக்கவும். ஆங்கில சொற்கள் இல்லை.",
                    'te': "దయచేసి తెలుగులో మాత్రమే సమాధానం ఇవ్వండి। ఇంగ్లీష్ పదాలు లేవు。",
                    'ml': "ദയവായി മലയാളത്തിൽ മാത്രം ഉത്തരം നൽകുക। ഇംഗ്ലീഷ് വാക്കുകളില്ला。",
                    'bn': "দয়া করে শুধুমাত্র বাংলায় উত্তর দিন। কোন ইংরেজি শব্দ নেই।",
                    'ur': "براہ کرم صرف اردو میں جواب دیں۔ کوئی انگریزی الفاظ نہیں۔",
                    'hi': "कृपया केवल हिंदी में उत्तर दें। कोई अंग्रेजी शब्द नहीं।",
                    'en': "Please respond only in English. No other languages.",
                    'or': "ଦୟାକରି କେବଳ ଓଡ଼ିଆରେ ଉତ୍ତର ଦିଅନ୍ତୁ। କୌଣସି ଇଂରାଜୀ ଶବ୍ଦ ନାହିଁ।"
                }

                enhanced_prompt = f"""
                    CRITICAL LANGUAGE INSTRUCTION: {language_prompts.get(target_language, '')}

                    You are Krishi Mitra, an agricultural expert helping Indian farmers with REAL-TIME DATA.

                    {context_string}

                    The farmer asked in {lang_name}: "{text}"

                    IMPORTANT INSTRUCTIONS:
                    1. Use the REAL-TIME weather and market data provided above to give CURRENT and RELEVANT advice
                    2. If weather data shows rain, advise accordingly. If it shows drought, give drought management tips
                    3. If market prices are provided, mention current rates and suggest optimal selling/buying decisions
                    4. Respond COMPLETELY in {lang_name} only
                    5. Give practical, actionable advice based on current conditions

                    Your response MUST:
                    - Be 100% in {lang_name}
                    - No English should use/
                    - Use the real-time data provided above
                    - Give specific advice based on current weather/market conditions
                    - Be practical and actionable for farmers
                    _ Dont use * symbol
                    START YOUR RESPONSE IN {lang_name} NOW:
                    """
                print(enhanced_prompt)
                print("------------------------------")

                response_text = self._generate_content(enhanced_prompt)

                # If the response contains English, try to translate it
# Skip translation check entirely for English target language
                if target_language != 'en' and response_text.isascii():
                    print(f"⚠️ Response is ASCII-only, attempting translation to {lang_name}")
                    try:
                        translated = self.translator.translate(response_text, dest=target_language)
                        return translated.text
                    except:
                        return self.get_emergency_response(target_language)

                return response_text

            except Exception as e:
                print(f"Error in enhanced text processing: {e}")
                return f"Error processing text query: {e}"

    def analyze_crop_image_with_context(self, image_path, query_text="", target_language="hi", location_info=None):
        """
        Enhanced image analysis.
    
        Key fix: the prompt is split into two explicit stages:
        Stage 1 — IDENTIFY the crop strictly from image evidence.
        Stage 2 — ADVISE based on that identification + weather/context.
        This prevents the LLM from skipping identification and jumping straight
        to generic crop advice.
        """
        try:
            if not os.path.exists(image_path):
                return f"Image file not found: {image_path}"
    
            contextual_data = self.get_contextual_data(query_text, location_info)
    
            lang_data = self.supported_languages.get(target_language, self.supported_languages['hi'])
            lang_name = lang_data['name']
    
            # Build weather context string
            weather_context = ""
            if 'weather' in contextual_data and contextual_data['weather'].get('success'):
                w  = contextual_data['weather']
                cur = w.get('current', {})
                weather_context = f"""
    CURRENT WEATHER (use this for advice):
    Temperature : {cur.get('temperature', 'N/A')}°C
    Humidity    : {cur.get('humidity', 'N/A')}%
    Condition   : {cur.get('description', 'N/A')}
    Wind speed  : {cur.get('wind_speed', 'N/A')} km/h
    """.strip()
    
            # Language instruction map
            language_prompts = {
                'hi': "Respond ENTIRELY in Hindi (हिन्दी). No English words.",
                'mr': "Respond ENTIRELY in Marathi (मराठी). No English words.",
                'pa': "Respond ENTIRELY in Punjabi (ਪੰਜਾਬੀ). No English words.",
                'kn': "Respond ENTIRELY in Kannada (ಕನ್ನಡ). No English words.",
                'ta': "Respond ENTIRELY in Tamil (தமிழ்). No English words.",
                'te': "Respond ENTIRELY in Telugu (తెలుగు). No English words.",
                'ml': "Respond ENTIRELY in Malayalam (മലയാളം). No English words.",
                'gu': "Respond ENTIRELY in Gujarati (ગુજરાતી). No English words.",
                'bn': "Respond ENTIRELY in Bengali (বাংলা). No English words.",
                'or': "Respond ENTIRELY in Odia (ଓଡ଼ିଆ). No English words.",
                'ur': "Respond ENTIRELY in Urdu (اردو). No English words.",
                'en': "Respond ENTIRELY in English.",
            }
            lang_instruction = language_prompts.get(target_language, "Respond ENTIRELY in English.")
    
            # ── Two-stage prompt ─────────────────────────────────────────────
            enhanced_prompt = f"""
    {lang_instruction}
    
    You are Krishi Mitra, an expert agricultural advisor for Indian farmers.
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STAGE 1 — CROP / PLANT IDENTIFICATION (mandatory first step)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Identify the crop or plant STRICTLY from the image evidence supplied.
    • If the image clearly shows a mango tree, say it is a mango tree.
    • If the image clearly shows a banana plant, say it is a banana plant.
    • Do NOT name any crop that is not visible in the image evidence.
    • Do NOT default to soybean, wheat, rice, or any other crop as a guess.
    • If the plant cannot be identified confidently, state that clearly.
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STAGE 2 — HEALTH ASSESSMENT & ADVICE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    After identifying the crop, provide:
    1. Visible health issues (spots, wilting, discolouration, pests)
    2. Impact of current weather on this specific crop
    3. Immediate actions the farmer should take
    4. Preventive measures
    
    {weather_context}
    
    ━━━━━━━━━━━
    FARMER SAID (in {lang_name}): "{query_text}"
    ━━━━━━━━━━━
    
    Now write your full response in {lang_name}. Do not use * symbols.
    """.strip()
    
            print("\n--- ENHANCED IMAGE PROMPT ---")
            print(enhanced_prompt)
            print("-----------------------------\n")
    
            response_text = self._generate_content(enhanced_prompt, image_path=image_path)
    
            # Fallback translation if response comes back as plain ASCII
            if target_language != 'en' and response_text.isascii():
                print(f"⚠️ Response is ASCII-only, attempting translation to {lang_name}")
                try:
                    from deep_translator import GoogleTranslator
                    translated = GoogleTranslator(source='auto', target=target_language).translate(response_text)
                    return translated
                except Exception:
                    return self.get_emergency_response(target_language)
    
            return response_text
    
        except Exception as e:
            print(f"Error in enhanced image analysis: {e}")
            return f"Error analyzing image: {e}"

    def get_emergency_response(self, language):
        """Emergency response when all else fails"""
        responses = {
            'mr': "माफ करा, तांत्रिक अडचण आहे. कृपया पुन्हा प्रयत्न करा.",
            'hi': "क्षमा करें, तकनीकी समस्या है। कृपया पुनः प्रयास करें।",
            'gu': "માફ કરશો, તકનીકી સમસ્યા છે. કૃપા કરીને ફરી પ્રયત્ન કરો.",
            'kn': "ಕ್ಷಮಿಸಿ, ತಾಂತ್ರಿಕ ಸಮಸ್ಯೆ ಇದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            'ta': "மன்னிக்கவும், தொழில்நுட்ப சிக்கல் உள்ளது. தயவு செய்து மீண்டும் முயற்சிக்கவும்.",
            'te': "క్షమించండి, సాంకేతిక సమస్య ఉంది. దయచేసి మళ్లీ ప్రయత్నించండి.",
            'pa': "ਮਾਫ ਕਰਨਾ, ਤਕਨੀਕੀ ਸਮੱਸਿਆ ਹੈ. ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ.",
            'ml': "ക്ഷമിക്കണം, സാങ്കേതിക പ്രശ്നമുണ്ട്. ദയവായി വീണ്ടും ശ്രമിക്കുക.",
            'bn': "ক্ষমা করবেন, প্রযুক্তিগত সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।",
            'ur': "معذرت، تکنیکی مسئلہ ہے۔ براہ کرم دوبارہ کوشش کریں۔",
            'en': "Technical issue. Please try again.",
            'or': "କ୍ଷମା କରନ୍ତୁ, ବୈଷୟିକ ସମସ୍ୟା ଅଛି। ଦୟାକରି ପୁଣି ଚେଷ୍ଟା କରନ୍ତୁ।"
        }
        return responses.get(language, "Technical issue. Please try again.")


class MarketPriceScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_commodity_prices(self, state="Maharashtra", district="Pune"):
        """Scrape commodity prices from government sources"""
        try:
            # Multiple sources for better reliability
            prices_data = []

            # Source 1: Try to get from agmarknet (government source)
            try:
                prices_data.extend(self._scrape_agmarknet(state, district))
            except Exception as e:
                print(f"Agmarknet scraping failed: {e}")

            # Source 2: Try alternative sources if available
            if not prices_data:
                prices_data = self._get_fallback_prices()

            return {
                "success": True,
                "data": prices_data,
                "location": f"{district}, {state}",
                "timestamp": datetime.now().isoformat(),
                "source": "Government Market Data"
            }

        except Exception as e:
            print(f"Error in get_commodity_prices: {e}")
            return {
                "success": False,
                "error": f"Failed to fetch market prices: {str(e)}",
                "fallback_data": self._get_static_sample_prices()
            }

    def _scrape_agmarknet(self, state, district):
        """Scrape from agmarknet.gov.in"""
        prices = []

        # This is a simplified example - actual implementation would need
        # to handle the complex JavaScript and form submissions of agmarknet
        base_url = "https://agmarknet.gov.in/"

        # For demo purposes, return sample data structure
        # In production, you'd implement actual scraping here
        sample_commodities = [
            {"name": "Rice (Common)", "variety": "FAQ", "unit": "Quintal",
             "min_price": 2000, "max_price": 2200, "modal_price": 2100},
            {"name": "Wheat", "variety": "Dara", "unit": "Quintal",
             "min_price": 2050, "max_price": 2150, "modal_price": 2100},
            {"name": "Onion", "variety": "Medium", "unit": "Quintal",
             "min_price": 1500, "max_price": 1800, "modal_price": 1650},
            {"name": "Potato", "variety": "Local", "unit": "Quintal",
             "min_price": 1200, "max_price": 1400, "modal_price": 1300},
            {"name": "Tomato", "variety": "Hybrid", "unit": "Quintal",
             "min_price": 800, "max_price": 1200, "modal_price": 1000}
        ]

        return sample_commodities

    def _get_fallback_prices(self):
        """Fallback method with sample data"""
        return [
            {"name": "Rice", "price_per_kg": 25, "change": "+2%", "unit": "Rs/Kg"},
            {"name": "Wheat", "price_per_kg": 22, "change": "-1%", "unit": "Rs/Kg"},
            {"name": "Onion", "price_per_kg": 18, "change": "+5%", "unit": "Rs/Kg"},
            {"name": "Potato", "price_per_kg": 15, "change": "+3%", "unit": "Rs/Kg"},
        ]

    def _get_static_sample_prices(self):
        """Static sample data when scraping fails"""
        return [
            {"commodity": "Rice", "price": "₹25/kg", "trend": "stable"},
            {"commodity": "Wheat", "price": "₹22/kg", "trend": "down"},
            {"commodity": "Onion", "price": "₹18/kg", "trend": "up"},
        ]

class WeatherScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_weather_data(self, city="Mumbai", state="Maharashtra"):
        """Get weather data from multiple sources"""
        try:
            weather_data = {}

            # Try multiple weather sources
            try:
                # Source 1: OpenWeatherMap (requires API key)
                #weather_data = self._get_openweather_data(city)
                raise Exception("Skipping OpenWeather for testing")
            except Exception as e:
                print(f"OpenWeather failed: {e}")
                try:
                    # Source 2: Weather.com scraping
                    weather_data = self._scrape_weather_com(city)
                except Exception as e2:
                    print(f"Weather.com scraping failed: {e2}")
                    # Fallback to sample data
                    weather_data = self._get_sample_weather_data(city)

            return {
                "success": True,
                "location": f"{city}, {state}",
                "current": weather_data.get("current", {}),
                "forecast": weather_data.get("forecast", []),
                "agricultural_advice": self._get_agricultural_weather_advice(weather_data),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error in get_weather_data: {e}")
            return {
                "success": False,
                "error": f"Failed to fetch weather data: {str(e)}",
                "fallback_data": self._get_sample_weather_data(city)
            }

    def _get_openweather_data(self, city):
        """Get data from OpenWeatherMap API"""
        # You'll need to sign up for a free API key at openweathermap.org
        API_KEY = "1cc9a339122ad5882e6df3b127ab43d8"  # Replace with actual key

        if API_KEY == "YOUR_OPENWEATHER_API_KEY":
            raise Exception("OpenWeather API key not configured")

        base_url = "http://api.openweathermap.org/data/2.5"

        # Current weather
        current_url = f"{base_url}/weather?q={city}&appid={API_KEY}&units=metric"
        current_response = requests.get(current_url, timeout=10)
        current_data = current_response.json()

        # 5-day forecast
        forecast_url = f"{base_url}/forecast?q={city}&appid={API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url, timeout=10)
        forecast_data = forecast_response.json()

        return {
            "current": {
                "temperature": current_data["main"]["temp"],
                "humidity": current_data["main"]["humidity"],
                "description": current_data["weather"][0]["description"],
                "wind_speed": current_data["wind"]["speed"],
                "pressure": current_data["main"]["pressure"]
            },
            "forecast": self._process_forecast_data(forecast_data["list"])
        }

    def _scrape_weather_com(self, city):
        """Scrape weather data from weather.com (example implementation)"""
        # This is a simplified example - actual weather.com scraping
        # would be more complex due to anti-scraping measures

        url = f"https://weather.com/weather/today/l/{city}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # This would need actual CSS selectors from weather.com
            # For demo, returning sample data
            return self._get_sample_weather_data(city)

        except Exception as e:
            print(f"Weather.com scraping error: {e}")
            return self._get_sample_weather_data(city)

    def _get_sample_weather_data(self, city):
        """Sample weather data for fallback"""
        return {
            "current": {
                "temperature": 28,
                "humidity": 65,
                "description": "Partly cloudy",
                "wind_speed": 12,
                "pressure": 1013
            },
            "forecast": [
                {"date": "Today", "temp_max": 32, "temp_min": 24, "description": "Sunny"},
                {"date": "Tomorrow", "temp_max": 30, "temp_min": 22, "description": "Cloudy"},
                {"date": "Day 3", "temp_max": 28, "temp_min": 20, "description": "Rain"},
            ]
        }

    def _process_forecast_data(self, forecast_list):
        """Process forecast data from API"""
        processed = []
        for item in forecast_list[:5]:  # Next 5 days
            processed.append({
                "date": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d"),
                "temp_max": item["main"]["temp_max"],
                "temp_min": item["main"]["temp_min"],
                "description": item["weather"][0]["description"],
                "humidity": item["main"]["humidity"]
            })
        return processed

    def _get_agricultural_weather_advice(self, weather_data):
        """Generate agricultural advice based on weather"""
        current = weather_data.get("current", {})
        temp = current.get("temperature", 25)
        humidity = current.get("humidity", 50)
        description = current.get("description", "").lower()

        advice = []

        if "rain" in description:
            advice.append("Good time for transplanting rice seedlings")
            advice.append("Ensure proper drainage in fields to prevent waterlogging")

        if temp > 35:
            advice.append("Increase irrigation frequency for crops")
            advice.append("Consider shade nets for sensitive crops")

        if humidity < 40:
            advice.append("Monitor crops for water stress")
            advice.append("Consider mulching to retain soil moisture")

        if humidity > 80:
            advice.append("Watch for fungal diseases in crops")
            advice.append("Ensure good air circulation in crop fields")

        return advice if advice else ["Weather conditions are favorable for most farming activities"]


# Flask API for integration with Flutter
# Flask API for integration with Flutter
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Initialize the agent
# Initialize the agent with multiple API keys
GROQ_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
agent = MultilingualFarmerAgent(groq_api_key=GROQ_KEY, hf_token=HF_TOKEN)

# ====== BASIC ROUTES ======
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Server is running"})

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get supported languages for dropdown"""
    languages = agent.get_supported_languages()
    return jsonify({"languages": languages})

# ====== ENHANCED ROUTES (FIXED) ======
@app.route('/api/process-query-enhanced', methods=['POST', 'OPTIONS'])  # Add OPTIONS for CORS
def process_query_enhanced():
    """Enhanced query processing with real-time data integration"""

    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    try:
        print("=" * 60)
        print("🚀 ENHANCED QUERY ENDPOINT HIT")
        print("=" * 60)

        data = request.json
        if not data:
            print("❌ No data provided in request")
            return jsonify({"error": "No data provided", "success": False}), 400

        language_name = data.get('language', 'English')
        language_code = agent.get_language_code_from_name(language_name)

        # Get location info (with defaults)
        location_info = {
            'city': data.get('city', 'Mumbai'),
            'state': data.get('state', 'Maharashtra'),
            'district': data.get('district', 'Pune')
        }

        print(f"📨 Enhanced processing for {language_name} -> {language_code}")
        print(f"📍 Location: {location_info}")

        # Handle image and audio files
        image_data = data.get('image')
        image_path = None
        if image_data:
            try:
                if ',' in image_data:
                    header, image_data = image_data.split(',', 1)
                image_bytes = base64.b64decode(image_data)
                timestamp = int(time.time() * 1000)
                image_path = os.path.join(tempfile.gettempdir(), f"image_{timestamp}.png")
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                print(f"✅ Image saved: {image_path}")
            except Exception as e:
                print(f"❌ Error processing image: {e}")
                return jsonify({"error": f"Error processing image: {e}", "success": False}), 400

        audio_data = data.get('audio')
        audio_path = None
        if audio_data:
            audio_path = agent.save_base64_audio_to_file(audio_data)
            if not audio_path:
                print("❌ Failed to process audio file")
                return jsonify({"error": "Failed to process audio file", "success": False}), 400

        # Use enhanced processing
        print("🔄 Starting enhanced processing...")
        result = agent.process_farmer_query_enhanced(
            image_path=image_path,
            audio_path=audio_path,
            language_code=language_code,
            location_info=location_info
        )

        # Cleanup files
        cleanup_files = [f for f in [image_path, audio_path] if f and os.path.exists(f)]
        for file_path in cleanup_files:
            try:
                os.unlink(file_path)
            except:
                pass

        print("✅ Enhanced processing completed successfully")
        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Enhanced processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {e}", "success": False}), 500


@app.route('/api/process-text-enhanced', methods=['POST', 'OPTIONS'])
def process_text_enhanced():
    """Enhanced text processing with real-time data"""

    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    try:
        print("=" * 60)
        print("📝 ENHANCED TEXT ENDPOINT HIT")
        print("=" * 60)

        data = request.json
        if not data:
            print("❌ No data provided")
            return jsonify({"error": "No data provided", "success": False}), 400

        language_name = data.get('language', 'English')
        text = data.get('text', '')
        language_code = agent.get_language_code_from_name(language_name)

        # Location info
        location_info = {
            'city': data.get('city', 'Mumbai'),
            'state': data.get('state', 'Maharashtra'),
            'district': data.get('district', 'Pune')
        }

        print(f"📝 Enhanced text processing: {text}")
        print(f"🌐 Language: {language_name} -> {language_code}")
        print(f"📍 Location: {location_info}")

        if not text.strip():
            print("❌ Empty text provided")
            return jsonify({"error": "Text cannot be empty", "success": False}), 400

        # Use enhanced text processing
        print("🔄 Starting text processing...")
        response_text = agent.process_text_query_with_context(
            text, language_code, location_info
        )

        print("✅ Text processing completed")
        return jsonify({
            "success": True,
            "query": text,
            "response": response_text,
            "language": language_code,
            "language_name": agent.get_language_name(language_code)
        }), 200

    except Exception as e:
        print(f"❌ Enhanced text processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {e}", "success": False}), 500


@app.route('/api/market-prices', methods=['GET'])
def get_market_prices():
    """Get real-time market prices"""
    try:
        print("💰 Market prices endpoint hit")

        # Get parameters from request
        state = request.args.get('state', 'Maharashtra')
        district = request.args.get('district', 'Pune')

        print(f"Fetching market prices for {district}, {state}")

        scraper = MarketPriceScraper()
        result = scraper.get_commodity_prices(state, district)

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Market prices API error: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to fetch market prices: {str(e)}"
        }), 500


@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Get weather data and agricultural advice"""
    try:
        print("🌦️ Weather endpoint hit")

        # Get parameters from request
        city = request.args.get('city', 'Mumbai')
        state = request.args.get('state', 'Maharashtra')

        print(f"Fetching weather data for {city}, {state}")

        scraper = WeatherScraper()
        result = scraper.get_weather_data(city, state)

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Weather API error: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to fetch weather data: {str(e)}"
        }), 500


# Test route to verify server is working
@app.route('/test', methods=['GET'])
def test_route():
    """Simple test route"""
    return jsonify({
        "message": "Server is working!",
        "routes": [str(rule) for rule in app.url_map.iter_rules()]
    }), 200


# ====== START SERVER ======
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌾 KRISHI MITRA SERVER STARTING...")
    print("=" * 60)
    print(f"🔗 Server URL: http://0.0.0.0:5000")
    print(f"📱 Flutter should connect to: http://192.168.1.110:5000")
    print("=" * 60)

    # Print audio processing status
    if PYDUB_AVAILABLE:
        print("✅ pydub available - advanced audio conversion enabled")
    else:
        print("⚠️ pydub/FFmpeg not available - using basic audio processing")

    print("\n📋 REGISTERED ROUTES:")
    print("-" * 60)
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
        print(f"  {rule.rule:40s} [{methods}]")
    print("=" * 60)
    print("\n🚀 Server is now running and waiting for requests...\n")

    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Changed to True for better error messages
        threaded=True,
        use_reloader=False  # Prevent double loading
    )