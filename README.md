# TweetSmith

A Flask web application that generates funny, viral-worthy tweets using AI and allows users to post them directly to their X (Twitter) account.

## Results at a glance

Measured on 15 fixed topics, run twice on Groq's free tier. "Before" is one draft per round with a small critic model; "after" is three drafts per round with a larger critic model.

| Metric | Before | After |
|---|---|---|
| Average time per tweet | 35 to 44 seconds | **11 to 19 seconds** (about 2.4x to 3.2x faster) |
| Average improvement rounds | 2.3 to 2.4 | **1.4 to 1.5** |
| Tweets over 280 characters | possible | **never** (enforced in code) |
| Automated tests | none | **73 passing** |

Full per-run numbers and caveats are in [Testing and evaluation](#testing-and-evaluation).

## Features

- 🤖 **AI-Powered Tweet Generation**: Uses Groq's LLM models to generate humorous tweets, writing several different candidates per round
- 🔄 **Iterative Optimization**: Automatically improves tweets based on AI feedback until they clear the quality bar
- 🧐 **Strict AI Critic**: A larger model scores every candidate from 1 to 10 and picks the best one
- 📏 **280-Character Safe**: The limit is enforced in code, not just requested in the prompt
- 🔐 **X OAuth Integration**: Secure login with X API for posting tweets
- 🎨 **Modern UI**: Clean, responsive interface with Twitter-like design
- 📱 **Mobile Friendly**: Works seamlessly on desktop and mobile devices
- 🧪 **Tested**: Automated tests run without any API keys or network access
- 📊 **Benchmark Script**: Compare pipeline settings with a fixed set of topics

## Setup

### Prerequisites

- Python 3.10 or higher
- Groq API key (the free tier works)
- X Developer Account with API access (only needed to post to X; see the note below)

> **Note on X API access:** generating tweets only needs the Groq key. Posting to X goes through the X API, which may require paid credits depending on X's current pricing. Check your developer portal before relying on it.

### Installation

1. **Clone the repository**

   ```
   git clone <repository-url>
   cd TweetSmith
   ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   # or using uv
   uv sync
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory with:

   ```
   GROQ_API_KEY=your_groq_api_key

   # Only needed for X login and posting:
   CLIENT_ID=your_x_client_id
   CLIENT_SECRET=your_x_client_secret
   ```

   In your X app settings, set the callback URI to exactly `http://127.0.0.1:5000/callback` and give the app read and write permissions.

4. **Run the application**

   ```
   python app.py
   ```

5. **Access the application**
   Open your browser and go to `http://127.0.0.1:5000`
   (use `127.0.0.1`, not `localhost`, because the OAuth redirect URI is set to `127.0.0.1`)

### Optional settings

All of these can go in `.env`. The defaults work out of the box.

| Variable | Default | What it does |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Model that writes and rewrites tweets |
| `GROQ_CRITIC_MODEL` | `openai/gpt-oss-120b` | Model that scores tweets |
| `NUM_CANDIDATES` | `3` | Tweets written per round |
| `APPROVAL_SCORE` | `7` | Score (1-10) at which a tweet is accepted |
| `GENERATOR_TEMPERATURE` | `1` | Creativity of the first draft |
| `OPTIMIZER_TEMPERATURE` | `0.8` | Creativity of the rewrites |
| `STRIP_TRAILING_HASHTAGS` | `1` | Set to `0` to keep hashtags at the end of tweets |

The maximum number of improvement rounds is `DEFAULT_MAX_ITERATION` (4) in `tweet_engine.py`.

**Style examples:** paste 3-8 of your favourite tweets into `style_examples.txt`, one per line. The AI uses them as a quality bar for both writing and judging, and does not copy them.

## Usage

1. **Generate Tweets**: Enter a topic and click "Generate Viral Tweet"
2. **Review Results**: See the generated tweet and, if it needed more than one round, the optimization history with the critic's scores
3. **Login to X**: Click "Connect X Account" to authenticate with your X account
4. **Post Tweets**: Once logged in, click "Post to X" to share your tweet

## API Endpoints

- `GET /` - Landing page
- `GET /tweet` - Tweet generator page
- `GET /login` - X OAuth login
- `GET /callback` - OAuth callback handler
- `GET /check-auth` - Check authentication status
- `GET /logout` - Logout user
- `GET /test-callback` - Check that the callback URL is reachable
- `POST /generate-tweet` - Generate AI tweet
- `POST /post-tweet` - Post tweet to X

`POST /generate-tweet` takes `{"topic": "..."}` (up to 200 characters) and returns `tweet`, `feedback`, `evaluation`, `iterations`, `char_count`, `tweet_history` and `feedback_history`.

## Architecture

- **Frontend**: HTML/CSS/JavaScript with modern UI design
- **Backend**: Flask web framework
- **AI**: Groq LLM models for tweet generation, scoring and optimization (`tweet_engine.py`)
- **Authentication**: X OAuth 2.0 for secure API access
- **Workflow**: LangGraph for iterative tweet optimization

How a tweet is made:

1. The writer model produces several different candidate tweets.
2. Candidates over 280 characters are rejected in code and never reach the critic.
3. The critic model scores each remaining candidate from 1 to 10 and picks the best.
4. If the best score reaches `APPROVAL_SCORE`, it is accepted. Otherwise the writer produces rewrites of the best tweet, and the previous best stays in the pool so a round can never make things worse.
5. The loop stops when a tweet is approved, when a round fails to beat the previous best, or after the maximum number of rounds. The best tweet from all rounds is returned.

The pipeline also cleans model output (intro lines, wrapping quotes, trailing hashtags), waits out short Groq rate limits, retries empty replies, and falls back to the writer model if the critic fails.

## Testing and evaluation

Run the automated tests (no API keys needed, the AI calls are faked):

```
pip install -r requirements.txt
pytest
```

Benchmark the pipeline on 15 fixed topics. A separate judge model with a fixed rubric scores each final tweet, so the pipeline's own critic does not grade itself. Needs `GROQ_API_KEY` only:

```
python evaluate.py --label improved
python evaluate.py --label baseline --candidates 1 --critic-model openai/gpt-oss-20b
python evaluate.py --compare baseline improved
```

Results are saved to `eval_results/`. A full run uses a large share of Groq's free daily token limit, so use `--topics 8` for a lighter run.

**Results from two of the author's runs on Groq's free tier** ("improved" is 3 candidates with the larger critic; "baseline" is 1 candidate with the small critic):

| | Run 1 baseline | Run 1 improved | Run 2 baseline | Run 2 improved |
|---|---|---|---|---|
| Avg seconds per tweet | 35.4 | 10.9 | 43.8 | 18.6 |
| Avg rounds | 2.27 | 1.38 | 2.36 | 1.54 |
| Avg judge score (1-10) | 6.91 | 7.38 | 7.29 | 6.85 |

The improved setup was consistently faster and needed fewer rounds. The judge score moved in opposite directions between runs, so the difference in tweet quality is within the noise of an LLM judge on about 14 topics. Some topics failed on rate limits and are excluded from the averages, so read the tweets yourself rather than trusting a single number.

## Security Notes

- Uses OAuth 2.0 for secure X API access
- Session-based authentication
- Environment variables for sensitive data (`.env` is git-ignored)
- CSRF protection through state parameter

Known limitations: this is a development setup. Login tokens are kept in memory and are lost when the server restarts, the Flask session key is regenerated on every start, debug mode is on, the redirect URI is hardcoded to localhost, and the OAuth routes print request details to the console. Fix these before deploying anywhere public.

## Development

The application consists of:

- `app.py` - Main Flask application and X OAuth routes
- `tweet_engine.py` - AI pipeline (generate, score, optimize)
- `tweet.html` - Tweet generator interface
- `index.html` - Landing page
- `evaluate.py` - Benchmark script
- `style_examples.txt` - Optional example tweets for the AI
- `tests/` and `conftest.py` - Automated tests
- `requirements.txt` - Python dependencies

## Troubleshooting

- **"AI models not available"**: Check your `GROQ_API_KEY` in `.env`
- **"model_not_found" or a 404 from Groq**: Groq retires models over time. Set `GROQ_MODEL` (and `GROQ_CRITIC_MODEL`) in `.env` to one listed at console.groq.com/docs/models
- **"Groq rate limit reached. Please try again in about N min"**: You hit Groq's free-tier token limit. Wait and retry
- **"Critic was unavailable" in the feedback**: The critic model failed, so the tweet was accepted without review. Check `GROQ_CRITIC_MODEL`
- **"Client ID not set"**: Ensure `CLIENT_ID` (and `CLIENT_SECRET`) are set in `.env`
- **Login issues**: Verify your X API credentials and that the redirect URI is exactly `http://127.0.0.1:5000/callback`

## License

This project is for educational purposes. Please respect X's API terms of service.
