from flask import Flask, request, jsonify, send_file, redirect, url_for, session
from dotenv import load_dotenv
import requests
import secrets
import os

load_dotenv(override=True)

import tweet_engine  # AI pipeline (generate -> evaluate -> optimize)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For sessions

# X API OAuth 2.0 settings
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = "http://127.0.0.1:5000/callback"  # Update for production
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"

# Check if required environment variables are set
if not CLIENT_ID:
    print("⚠️  WARNING: CLIENT_ID not found in environment variables")
    print("   Please set CLIENT_ID in your .env file for X OAuth functionality")

if not CLIENT_SECRET:
    print("⚠️  WARNING: CLIENT_SECRET not found in environment variables")
    print("   Please set CLIENT_SECRET in your .env file for X OAuth functionality")

# In-memory storage for user tokens (use database in production)
user_tokens = {}

# Groq models (a failure here no longer crashes the app on startup)
tweet_engine.init_models()

# OAuth 2.0 Routes
@app.route('/login')
def login():
    print("🔍 Login route accessed!")
    
    if not CLIENT_ID:
        return jsonify({'error': 'Client ID not set'}), 500

    # Generate a random state for security
    state = secrets.token_urlsafe(16)
    session['state'] = state
    session['user_id'] = secrets.token_urlsafe(16)  # Unique user ID
    
    # Generate PKCE parameters
    code_verifier = secrets.token_urlsafe(32)
    import hashlib
    import base64
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).decode().rstrip('=')

    # Store code_verifier in session
    session['code_verifier'] = code_verifier

    # Redirect to X login
    auth_params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'tweet.read tweet.write users.read offline.access',
        'state': state,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256'
    }
    auth_url = f"{AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in auth_params.items())}"
    
    print(f"Redirecting to: {auth_url}")
    print(f"State: {state}")
    print(f"Redirect URI: {REDIRECT_URI}")
    print(f"Code challenge: {code_challenge}")
    
    return redirect(auth_url)

@app.route('/callback')
def callback():
    print("🔍 Callback received!")
    print(f"URL args: {dict(request.args)}")
    
    code = request.args.get('code')
    state = request.args.get('state')
    error = request.args.get('error')
    error_description = request.args.get('error_description')
    
    saved_state = session.get('state')
    
    print(f"Code: {code}")
    print(f"State: {state}")
    print(f"Saved state: {saved_state}")
    print(f"Error: {error}")
    print(f"Error description: {error_description}")

    if error:
        return jsonify({'error': f'OAuth error: {error}', 'description': error_description}), 400

    if not code or state != saved_state:
        return jsonify({'error': 'Invalid login attempt'}), 400

    # Exchange code for access token
    code_verifier = session.get('code_verifier')
    if not code_verifier:
        return jsonify({'error': 'Code verifier not found in session'}), 400
        
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI,
        'code_verifier': code_verifier
    }
    
    try:
        # Use basic auth with client_id and client_secret
        import base64
        credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': f'Basic {encoded_credentials}'
        }
        
        print(f"Token data being sent: {token_data}")
        print(f"Headers: {headers}")
        
        response = requests.post(TOKEN_URL, data=token_data, headers=headers)
        print(f"Token response status: {response.status_code}")
        print(f"Token response: {response.text}")
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to get token', 'details': response.text}), 500

        token_response = response.json()
        user_tokens[session['user_id']] = {
            'access_token': token_response['access_token'],
            'refresh_token': token_response.get('refresh_token')
        }
        return redirect(url_for('tweet_page'))
    except Exception as e:
        return jsonify({'error': f'Authentication failed: {str(e)}'}), 500

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    if user_id and user_id in user_tokens:
        del user_tokens[user_id]
    session.clear()
    return redirect(url_for('landing_page'))

@app.route('/')
def landing_page():
    return send_file('index.html')

@app.route('/tweet')
def tweet_page():
    return send_file('tweet.html')

@app.route('/check-auth')
def check_auth():
    user_id = session.get('user_id')
    is_authenticated = user_id is not None and user_id in user_tokens
    return jsonify({'authenticated': is_authenticated})

@app.route('/test-callback')
def test_callback():
    """Test endpoint to verify callback URL is accessible"""
    return jsonify({'message': 'Callback URL is working!', 'timestamp': 'test'})

@app.route('/generate-tweet', methods=['POST'])
def generate_tweet_api():
    data = request.get_json(silent=True) or {}
    topic = data.get('topic')

    if not isinstance(topic, str) or not topic.strip():
        return jsonify({'error': 'No topic provided'}), 400
    topic = topic.strip()
    if len(topic) > tweet_engine.MAX_TOPIC_CHARS:
        return jsonify({'error': f'Topic is too long (max {tweet_engine.MAX_TOPIC_CHARS} characters)'}), 400

    # Check if LLM models are available
    if not tweet_engine.models_ready():
        return jsonify({'error': 'AI models not available. Please check your GROQ_API_KEY configuration.'}), 500

    try:
        result = tweet_engine.run_workflow(topic)
        tweet = result['tweet']
        return jsonify({
            'tweet': tweet,
            'feedback': result.get('feedback', ''),
            'evaluation': result.get('evaluation', ''),
            'iterations': result.get('iteration', 1),
            'char_count': len(tweet),
            'tweet_history': result.get('tweet_history', []),
            'feedback_history': result.get('feedback_history', []),
        })
    except Exception as e:
        return jsonify({'error': f'Failed to generate tweet: {str(e)}'}), 500

@app.route('/post-tweet', methods=['POST'])
def post_tweet_api():
    data = request.get_json()
    tweet = data.get('tweet')
    user_id = session.get('user_id')

    if not tweet:
        return jsonify({'error': 'No tweet provided'}), 400
    if not user_id or user_id not in user_tokens:
        return jsonify({'error': 'Please log in with X'}), 401

    access_token = user_tokens[user_id]['access_token']
    headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
    payload = {'text': tweet}
    response = requests.post('https://api.twitter.com/2/tweets', json=payload, headers=headers)

    if response.status_code == 201:
        tweet_id = response.json().get('data', {}).get('id')
        return jsonify({'success': True, 'tweet_id': tweet_id})
    return jsonify({'error': 'Failed to post tweet', 'details': response.text}), 500

if __name__ == '__main__':
    app.run(debug=True)
