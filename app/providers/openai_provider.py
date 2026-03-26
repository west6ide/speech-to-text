import logging
import time

from openai import APIConnectionError, APIError, AsyncOpenAI, AuthenticationError, RateLimitError

from app.config import Settings
from app.errors import TTSConfigurationError, TTSProviderError
from app.models import AudioTranscriptionResponse, SUPPORTED_AUDIO_FORMATS, TTSRequest, TTSResponse
from app.providers.base import BaseTTSProvider


class OpenAITTSProvider(BaseTTSProvider):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise TTSConfigurationError("OPENAI_API_KEY is not configured")

        client_kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._settings = settings
        self._logger = logging.getLogger("app.providers.openai")
        self._client = AsyncOpenAI(
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            **client_kwargs,
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        voice = request.voice or self._settings.openai_tts_voice
        model = request.model or self._settings.openai_tts_model
        started_at = time.perf_counter()

        try:
            request_payload = {
                "model": model,
                "input": request.text,
                "format": request.format,
                "speed": request.speed,
            }
            if voice:
                request_payload["voice"] = voice

            self._logger.info(
                "tts_request_started model=%s voice=%s format=%s text_chars=%s",
                model,
                voice or "-",
                request.format,
                len(request.text),
            )
            response = await self._client.audio.speech.create(**request_payload)
            audio_bytes = response.read()
        except AuthenticationError as exc:
            raise TTSProviderError("Authentication with TTS provider failed") from exc
        except RateLimitError as exc:
            raise TTSProviderError("TTS provider rate limit exceeded") from exc
        except APIConnectionError as exc:
            raise TTSProviderError("TTS provider is unreachable") from exc
        except APIError as exc:
            raise TTSProviderError("TTS provider returned an API error") from exc
        except Exception as exc:
            raise TTSProviderError("Unexpected provider failure") from exc

        if not audio_bytes:
            raise TTSProviderError("TTS provider returned empty audio")

        self._logger.info(
            "tts_request_finished model=%s duration_ms=%s audio_bytes=%s",
            model,
            round((time.perf_counter() - started_at) * 1000, 2),
            len(audio_bytes),
        )

        return TTSResponse(
            content=audio_bytes,
            content_type=SUPPORTED_AUDIO_FORMATS[request.format],
            filename=f"speech.{request.format}",
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None = None,
        model: str | None = None,
        report_language: str = "ru",
    ) -> AudioTranscriptionResponse:
        transcription_model = model or self._settings.openai_stt_model
        started_at = time.perf_counter()

        try:
            self._logger.info(
                "stt_request_started model=%s filename=%s content_type=%s audio_bytes=%s",
                transcription_model,
                filename,
                content_type or "application/octet-stream",
                len(audio_bytes),
            )
            response = await self._client.audio.transcriptions.create(
                model=transcription_model,
                file=(filename, audio_bytes, content_type or "application/octet-stream"),
                language=report_language,
            )
        except AuthenticationError as exc:
            raise TTSProviderError("Authentication with STT provider failed") from exc
        except RateLimitError as exc:
            raise TTSProviderError("STT provider rate limit exceeded") from exc
        except APIConnectionError as exc:
            raise TTSProviderError("STT provider is unreachable") from exc
        except APIError as exc:
            raise TTSProviderError("STT provider returned an API error") from exc
        except Exception as exc:
            raise TTSProviderError("Unexpected STT provider failure") from exc

        text = getattr(response, "text", None)
        if text is None and isinstance(response, str):
            text = response

        if not text or not str(text).strip():
            raise TTSProviderError("STT provider returned empty transcription")

        self._logger.info(
            "stt_request_finished model=%s duration_ms=%s transcript_chars=%s",
            transcription_model,
            round((time.perf_counter() - started_at) * 1000, 2),
            len(str(text).strip()),
        )

        return AudioTranscriptionResponse(
            text=str(text).strip(),
            raw_text=str(text).strip(),
            corrected_text=str(text).strip(),
            model=transcription_model,
            filename=filename,
            duration_seconds=None,
            chunk_count=1,
        )
