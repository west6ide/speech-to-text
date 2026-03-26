# Text-to-Speech Service

HTTP API for text-to-speech generation and text quality analysis through an OpenAI-compatible provider.

## Recommended Models

- TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Analysis: `openai/gpt-oss-120b`

`openai/whisper-large-v3` and `openai/whisper-large-v3-turbo` are speech-to-text models, not TTS models.

## Features

- `POST /v1/text-to-speech` for audio generation
- `POST /v1/text-analysis` for pre-TTS text quality analysis
- `POST /v1/text-to-speech/report` for combined output: analysis + audio
- `POST /v1/meeting/analyze` for meeting transcription, cleanup, summary, decisions, and action items
- Validation for text, format, speed, and voice
- Provider error handling with stable HTTP responses

## Configuration

Example `.env`:

```env
TTS_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
OPENAI_TTS_VOICE=
OPENAI_ANALYSIS_MODEL=openai/gpt-oss-120b
OPENAI_BASE_URL=https://llm.nitec.kz/v1
OPENAI_TIMEOUT_SECONDS=120
OPENAI_MAX_RETRIES=0
LOG_LEVEL=INFO
ANALYSIS_RETRY_ON_INVALID_JSON=false
FFMPEG_PATH=ffmpeg
FFPROBE_PATH=ffprobe
STT_CHUNK_DURATION_SECONDS=300
STT_MAX_PARALLEL_CHUNKS=2
PORT=8000
MAX_TEXT_LENGTH=5000
```

Important: `OPENAI_BASE_URL` must be the base `/v1` URL, not a full endpoint like `/v1/chat/completions`.

## Run

```bash
pip install -e .[dev]
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

Build image:

```bash
docker build -t meeting-ai-service .
```

Run container:

```bash
docker run --rm -p 8000:8000 --env-file .env meeting-ai-service
```

Or with Compose:

```bash
docker compose up --build -d
```

Stop:

```bash
docker compose down
```

Check logs:

```bash
docker compose logs -f
```

## Example Requests

TTS request:

```json
{
  "text": "Привет. Это тест синтеза речи.",
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "format": "mp3",
  "speed": 1.0
}
```

Analysis request:

```json
{
  "text": "Привет привет... это это тест 123 / api"
}
```

Meeting analysis request:

```bash
curl.exe -X POST "http://127.0.0.1:8000/v1/meeting/analyze" -F "report_language=ru" -F "file=@C:\path\to\meeting.wav;type=audio/wav"
```

Meeting analysis response contains:

- `transcription.raw_text`
- `transcription.corrected_text`
- `analysis`
- `summary.short_summary`
- `summary.detailed_summary`
- `summary.topics`
- `summary.decisions`
- `summary.action_items`
- `summary.open_questions`

Healthcheck:

```bash
curl http://127.0.0.1:8000/health
```

## Notes

- If your provider supports only `/chat/completions` and does not implement `audio/speech`, the TTS endpoint will not work with the current client.
- Text analysis combines an LLM response with local heuristics.
- `speech-to-text/report` is slower than `speech-to-text` because it does two external operations in sequence: transcription first, analysis second.
- Logs now show request timing, STT timing, analysis timing, and whether fallback analysis was used.
- Long audio is handled by `ffmpeg`: the service normalizes input to mono 16 kHz WAV, splits it into chunks, transcribes chunks, and merges them into one response.
- For hour-long recordings, start with `STT_CHUNK_DURATION_SECONDS=300` and `STT_MAX_PARALLEL_CHUNKS=2`.
