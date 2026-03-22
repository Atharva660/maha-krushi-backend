import os
import sys
from groq import Groq
from huggingface_hub import InferenceClient
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
# import pygame
from PIL import Image
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
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None
        self.hf_client = InferenceClient(token=self.hf_token) if self.hf_token else None
        self.recognizer = sr.Recognizer()
        print("✅ API Clients initialized (100% FREE)")
        if self.groq_client:
            print("  - Groq: Active (Text queries)")
        if self.hf_client:
            print("  - Hugging Face: Active (Image analysis)")

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

        self.language_name_to_code = {
            'English': 'en', 'हिन्दी': 'hi', 'मराठी': 'mr', 'ਪੰਜਾਬੀ': 'pa',
            'ಕನ್ನಡ': 'kn', 'தமிழ்': 'ta', 'తెలుగు': 'te', 'മലയാളം': 'ml',
            'ગુજરાતી': 'gu', 'বাংলা': 'bn', 'ଓଡ଼ିଆ': 'or', 'اردو': 'ur'
        }

    def _generate_content(self, prompt, image_path=None, max_retries=3):
        """Generate content with Groq (text) and Hugging Face (images)."""

        if image_path and self.hf_client:
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                # Simplified image route: get caption and pass to text model
                caption = self.hf_client.image_to_text(image_bytes, model='Salesforce/blip-image-captioning-large')
                combined_prompt = f"Image description: {caption}

{prompt}"
                if self.groq_client:
                    response = self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are Krishi Mitra, an expert agricultural advisor."},
                            {"role": "user", "content": combined_prompt}
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.7,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Image pipeline failed: {e}")

        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are Krishi Mitra, an expert agricultural advisor."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Groq text failed: {e}")

        return "सेवा अस्थायी रूप से अनुपलब्ध है। कृपया बाद में प्रयास करें। (Service temporarily unavailable)"
