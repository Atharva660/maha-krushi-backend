from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health_check():
    print("Health check requested")
    return {'status': 'ok', 'message': 'Server is running'}, 200

@app.route('/api/languages', methods=['GET'])
def get_languages():
    print("Languages requested")
    languages = [
        {'code': 'hi', 'name': 'Hindi'},
        {'code': 'en', 'name': 'English'},
        {'code': 'pa', 'name': 'Punjabi'}
    ]
    return {'languages': languages}, 200

@app.route('/api/process-query', methods=['POST'])
def process_query():
    try:
        print("Process query requested")
        data = request.get_json()
        print(f"Received data keys: {list(data.keys()) if data else 'None'}")

        if not data:
            return {'success': False, 'error': 'No data received'}, 400

        language = data.get('language', 'en')
        has_image = 'image' in data
        has_audio = 'audio' in data

        print(f"Language: {language}")
        print(f"Has image: {has_image}")
        print(f"Has audio: {has_audio}")

        # Simulate processing
        response_data = {
            'success': True,
            'query': 'Sample query from audio' if has_audio else None,
            'analysis': f'Analysis complete for {language}. Image: {has_image}, Audio: {has_audio}'
        }

        return response_data, 200

    except Exception as e:
        print(f"Error processing query: {e}")
        return {'success': False, 'error': str(e)}, 500

@app.route('/api/process-text', methods=['POST'])
def process_text():
    try:
        print("Process text requested")
        data = request.get_json()

        if not data:
            return {'success': False, 'error': 'No data received'}, 400

        text = data.get('text', '')
        language = data.get('language', 'en')

        print(f"Text: {text}")
        print(f"Language: {language}")

        response_data = {
            'success': True,
            'response': f'I received your message in {language}: "{text}". This is a test response.'
        }

        return response_data, 200

    except Exception as e:
        print(f"Error processing text: {e}")
        return {'success': False, 'error': str(e)}, 500

@app.route('/', methods=['GET'])
def root():
    return {'message': 'Agricultural AI Assistant API', 'status': 'running'}, 200

if __name__ == '__main__':
    print("Starting server...")
    print("Available endpoints:")
    print("- GET  /health")
    print("- GET  /api/languages")
    print("- POST /api/process-query")
    print("- POST /api/process-text")
    print("")

    # Run on all interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)