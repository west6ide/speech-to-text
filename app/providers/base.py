from abc import ABC, abstractmethod

from app.models import AudioTranscriptionResponse, TTSRequest, TTSResponse


class BaseTTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        raise NotImplementedError

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None = None,
        model: str | None = None,
        report_language: str = "ru",
    ) -> AudioTranscriptionResponse:
        raise NotImplementedError
