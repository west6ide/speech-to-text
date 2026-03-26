import asyncio
import tempfile
from pathlib import Path

from fastapi import HTTPException

from app.config import Settings
from app.errors import TTSProviderError
from app.models import (
    AnalysisIssue,
    AudioTranscriptionResponse,
    TTSRequest,
    TTSResponse,
    TextAnalysisRequest,
    TextAnalysisResult,
)
from app.service import TTSService


class DummyProvider:
    def __init__(
        self,
        response: TTSResponse | None = None,
        error: Exception | None = None,
        transcription: AudioTranscriptionResponse | None = None,
        transcriptions_by_filename: dict[str, AudioTranscriptionResponse] | None = None,
    ):
        self._response = response
        self._error = error
        self._transcription = transcription
        self._transcriptions_by_filename = transcriptions_by_filename or {}

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        if self._error:
            raise self._error
        return self._response  # type: ignore[return-value]

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None = None,
        model: str | None = None,
        report_language: str = "ru",
    ):
        if self._error:
            raise self._error
        if filename in self._transcriptions_by_filename:
            return self._transcriptions_by_filename[filename]
        return self._transcription  # type: ignore[return-value]


class DummyPreparedAudio:
    def __init__(self, chunks: list[Path], duration_seconds: float, variants=None, channel_count: int = 1):
        self.chunks = chunks
        self.duration_seconds = duration_seconds
        self.variants = variants or [type("Variant", (), {"name": "mono", "chunks": chunks})()]
        self.channel_count = channel_count
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class DummyPreprocessor:
    def __init__(self, prepared_audio: DummyPreparedAudio):
        self._prepared_audio = prepared_audio

    def prepare(self, audio_bytes: bytes, filename: str) -> DummyPreparedAudio:
        return self._prepared_audio


class DummyAnalyzer:
    def __init__(self, response: TextAnalysisResult | None = None, error: Exception | None = None):
        self._response = response
        self._error = error

    async def analyze(self, request: TextAnalysisRequest) -> TextAnalysisResult:
        if self._error:
            raise self._error
        return self._response  # type: ignore[return-value]

    def correct_text(self, text: str) -> str:
        normalized = text.strip()
        if normalized and normalized[-1] not in ".!?":
            return f"{normalized}."
        return normalized


class DummyMeetingAnalyzer:
    def __init__(self, refined_text: str = "Hello world.", should_refine: bool = True):
        self._refined_text = refined_text
        self._should_refine = should_refine

    def should_refine_transcript(self, text: str) -> bool:
        return self._should_refine

    async def refine_transcript(self, text: str, report_language: str) -> str:
        return self._refined_text

    async def summarize_meeting(self, text: str, report_language: str):
        from app.models import MeetingSummary

        return MeetingSummary(
            short_summary="Short summary.",
            detailed_summary="Detailed summary.",
            topics=["topic"],
            decisions=[],
            action_items=[],
            open_questions=[],
            risks=[],
        )

    def fallback_summary(self, text: str, report_language: str):
        from app.models import MeetingSummary

        return MeetingSummary(
            short_summary="Fallback summary.",
            detailed_summary="Fallback detailed summary.",
            topics=[],
            decisions=[],
            action_items=[],
            open_questions=[],
            risks=[],
        )


def build_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_TTS_MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        OPENAI_STT_MODEL="openai/whisper-large-v3-turbo",
        MAX_TEXT_LENGTH=100,
        STT_MAX_PARALLEL_CHUNKS=2,
    )


def test_service_returns_audio():
    settings = build_settings()
    provider = DummyProvider(
        response=TTSResponse(content=b"audio", content_type="audio/mpeg", filename="speech.mp3")
    )
    service = TTSService(settings, provider=provider)

    result = asyncio.run(service.synthesize(TTSRequest(text="hello", format="mp3")))

    assert result.content == b"audio"
    assert result.content_type == "audio/mpeg"


def test_service_returns_analysis():
    settings = build_settings()
    provider = DummyProvider(
        response=TTSResponse(content=b"audio", content_type="audio/mpeg", filename="speech.mp3")
    )
    service = TTSService(settings, provider=provider)
    service._analyzer = DummyAnalyzer(
        response=TextAnalysisResult(
            summary="Text is mostly clear.",
            overall_quality="good",
            coherence_score=90,
            wording_score=88,
            meaning_score=91,
            pronunciation_risk_score=15,
            corrected_text="Hello world.",
            issues=[
                AnalysisIssue(
                    severity="low",
                    category="style",
                    fragment="hello",
                    explanation="Neutral sample.",
                    suggestion="No action needed.",
                )
            ],
            recommendations=["Keep punctuation natural."],
        )
    )

    result = asyncio.run(service.analyze_text(TextAnalysisRequest(text="hello world")))

    assert result.overall_quality == "good"
    assert result.coherence_score == 90


def test_service_rejects_too_long_text():
    settings = build_settings()
    provider = DummyProvider(
        response=TTSResponse(content=b"audio", content_type="audio/mpeg", filename="speech.mp3")
    )
    service = TTSService(settings, provider=provider)

    try:
        asyncio.run(service.synthesize(TTSRequest(text="a" * 101, format="mp3")))
    except HTTPException as exc:
        assert exc.status_code == 422
        assert "MAX_TEXT_LENGTH" in exc.detail
    else:
        raise AssertionError("Expected HTTPException")


def test_service_maps_provider_failure():
    settings = build_settings()
    provider = DummyProvider(error=TTSProviderError("provider failed"))
    service = TTSService(settings, provider=provider)

    try:
        asyncio.run(service.synthesize(TTSRequest(text="hello", format="mp3")))
    except HTTPException as exc:
        assert exc.status_code == 502
        assert exc.detail == "provider failed"
    else:
        raise AssertionError("Expected HTTPException")


def test_service_builds_report():
    settings = build_settings()
    provider = DummyProvider(
        response=TTSResponse(content=b"audio", content_type="audio/mpeg", filename="speech.mp3")
    )
    service = TTSService(settings, provider=provider)
    service._analyzer = DummyAnalyzer(
        response=TextAnalysisResult(
            summary="Clear text.",
            overall_quality="good",
            coherence_score=95,
            wording_score=93,
            meaning_score=94,
            pronunciation_risk_score=10,
            corrected_text="Hello world.",
            issues=[],
            recommendations=["Looks good."],
        )
    )

    result = asyncio.run(
        service.synthesize_with_report(TTSRequest(text="hello world", format="mp3"))
    )

    assert result.filename == "speech.mp3"
    assert result.audio_base64 == "YXVkaW8="
    assert result.analysis.overall_quality == "good"


def test_service_transcribes_audio():
    settings = build_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk = Path(temp_dir) / "chunk_0001.wav"
        chunk.write_bytes(b"chunk-1")
        provider = DummyProvider(
        transcription=AudioTranscriptionResponse(
            text="hello world",
            raw_text="hello world",
            corrected_text="Hello world.",
            model="openai/whisper-large-v3-turbo",
            filename="chunk_0001.wav",
        )
        )
        service = TTSService(settings, provider=provider)
        service._preprocessor = DummyPreprocessor(DummyPreparedAudio(chunks=[chunk], duration_seconds=12.0))
        service._meeting_analyzer = DummyMeetingAnalyzer(refined_text="Hello world.", should_refine=True)

        result = asyncio.run(
            service.transcribe_audio(
                audio_bytes=b"fake-audio",
                filename="sample.mp3",
                content_type="audio/mpeg",
            )
        )

        assert result.text == "hello world"
        assert result.filename == "sample.mp3"
        assert result.chunk_count == 1
        assert result.corrected_text == "Hello world."


def test_service_transcribes_and_analyzes_audio():
    settings = build_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk = Path(temp_dir) / "chunk_0001.wav"
        chunk.write_bytes(b"chunk-1")
        provider = DummyProvider(
        transcription=AudioTranscriptionResponse(
            text="hello world",
            raw_text="hello world",
            corrected_text="Hello world.",
            model="openai/whisper-large-v3-turbo",
            filename="chunk_0001.wav",
        )
        )
        service = TTSService(settings, provider=provider)
        service._preprocessor = DummyPreprocessor(DummyPreparedAudio(chunks=[chunk], duration_seconds=12.0))
        service._meeting_analyzer = DummyMeetingAnalyzer(refined_text="Hello world.", should_refine=True)
        service._analyzer = DummyAnalyzer(
        response=TextAnalysisResult(
            summary="Clear text.",
            overall_quality="good",
            coherence_score=95,
            wording_score=93,
            meaning_score=94,
            pronunciation_risk_score=10,
            corrected_text="Hello world.",
            issues=[],
            recommendations=["Looks good."],
            )
        )

        result = asyncio.run(
            service.transcribe_and_analyze_audio(
                audio_bytes=b"fake-audio",
                filename="sample.mp3",
                content_type="audio/mpeg",
            )
        )

        assert result.transcription.text == "hello world"
        assert result.analysis.overall_quality == "good"


def test_service_transcribes_long_audio_in_chunks():
    settings = build_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk1 = Path(temp_dir) / "chunk_0001.wav"
        chunk2 = Path(temp_dir) / "chunk_0002.wav"
        chunk1.write_bytes(b"chunk-1")
        chunk2.write_bytes(b"chunk-2")

        provider = DummyProvider(
            transcriptions_by_filename={
                "chunk_0001.wav": AudioTranscriptionResponse(
                    text="first part",
                    raw_text="first part",
                    corrected_text="First part.",
                    model="openai/whisper-large-v3-turbo",
                    filename="chunk_0001.wav",
                ),
                "chunk_0002.wav": AudioTranscriptionResponse(
                    text="second part",
                    raw_text="second part",
                    corrected_text="Second part.",
                    model="openai/whisper-large-v3-turbo",
                    filename="chunk_0002.wav",
                ),
            }
        )
        service = TTSService(settings, provider=provider)
        prepared_audio = DummyPreparedAudio(chunks=[chunk1, chunk2], duration_seconds=3665.0)
        service._preprocessor = DummyPreprocessor(prepared_audio)
        service._meeting_analyzer = DummyMeetingAnalyzer(refined_text="First part second part.", should_refine=True)

        result = asyncio.run(
            service.transcribe_audio(
                audio_bytes=b"fake-audio",
                filename="meeting.wav",
                content_type="audio/wav",
            )
        )

        assert result.text == "first part second part"
        assert result.chunk_count == 2
        assert result.duration_seconds == 3665.0
        assert prepared_audio.cleaned is True


def test_service_prefers_cleaner_stereo_channel():
    settings = build_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        left_chunk = Path(temp_dir) / "left_chunk_0001.wav"
        right_chunk = Path(temp_dir) / "right_chunk_0001.wav"
        mono_chunk = Path(temp_dir) / "mono_chunk_0001.wav"
        phase_chunk = Path(temp_dir) / "phase_fixed_chunk_0001.wav"
        reverse_phase_chunk = Path(temp_dir) / "phase_fixed_reverse_chunk_0001.wav"
        for path in (left_chunk, right_chunk, mono_chunk, phase_chunk, reverse_phase_chunk):
            path.write_bytes(b"chunk")

        provider = DummyProvider(
            transcriptions_by_filename={
                "left_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Сәлеметсіз бе мен қоңырау шалып тұрмын",
                    raw_text="Сәлеметсіз бе мен қоңырау шалып тұрмын",
                    corrected_text="Сәлеметсіз бе мен қоңырау шалып тұрмын.",
                    model="openai/whisper-large-v3-turbo",
                    filename="left_chunk_0001.wav",
                ),
                "right_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Қалайсын Қалайсын Қалайсын",
                    raw_text="Қалайсын Қалайсын Қалайсын",
                    corrected_text="Қалайсын Қалайсын Қалайсын.",
                    model="openai/whisper-large-v3-turbo",
                    filename="right_chunk_0001.wav",
                ),
                "mono_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Қалайсын Қалайсын Kísi",
                    raw_text="Қалайсын Қалайсын Kísi",
                    corrected_text="Қалайсын Қалайсын Kísi.",
                    model="openai/whisper-large-v3-turbo",
                    filename="mono_chunk_0001.wav",
                ),
                "phase_fixed_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Сәлеметсіз бе мен қоңырау шалып тұрмын",
                    raw_text="Сәлеметсіз бе мен қоңырау шалып тұрмын",
                    corrected_text="Сәлеметсіз бе мен қоңырау шалып тұрмын.",
                    model="openai/whisper-large-v3-turbo",
                    filename="phase_fixed_chunk_0001.wav",
                ),
                "phase_fixed_reverse_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Thank you",
                    raw_text="Thank you",
                    corrected_text="Thank you.",
                    model="openai/whisper-large-v3-turbo",
                    filename="phase_fixed_reverse_chunk_0001.wav",
                ),
            }
        )
        service = TTSService(settings, provider=provider)
        variants = [
            type("Variant", (), {"name": "left", "chunks": [left_chunk]})(),
            type("Variant", (), {"name": "right", "chunks": [right_chunk]})(),
            type("Variant", (), {"name": "phase_fixed", "chunks": [phase_chunk]})(),
            type("Variant", (), {"name": "phase_fixed_reverse", "chunks": [reverse_phase_chunk]})(),
            type("Variant", (), {"name": "mono", "chunks": [mono_chunk]})(),
        ]
        service._preprocessor = DummyPreprocessor(
            DummyPreparedAudio(
                chunks=[left_chunk, right_chunk, mono_chunk, phase_chunk, reverse_phase_chunk],
                duration_seconds=12.0,
                variants=variants,
                channel_count=2,
            )
        )
        service._meeting_analyzer = DummyMeetingAnalyzer(refined_text="Сәлеметсіз бе мен қоңырау шалып тұрмын.", should_refine=True)

        result = asyncio.run(
            service.transcribe_audio(
                audio_bytes=b"fake-audio",
                filename="call.wav",
                content_type="audio/wave",
                report_language='"kk"',
            )
        )

        assert result.text == "Сәлеметсіз бе мен қоңырау шалып тұрмын"
        assert result.model == "openai/whisper-large-v3-turbo"


def test_service_merges_distinct_stereo_channels():
    settings = build_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        left_chunk = Path(temp_dir) / "left_chunk_0001.wav"
        right_chunk = Path(temp_dir) / "right_chunk_0001.wav"
        mono_chunk = Path(temp_dir) / "mono_chunk_0001.wav"
        phase_chunk = Path(temp_dir) / "phase_fixed_chunk_0001.wav"
        reverse_phase_chunk = Path(temp_dir) / "phase_fixed_reverse_chunk_0001.wav"
        for path in (left_chunk, right_chunk, mono_chunk, phase_chunk, reverse_phase_chunk):
            path.write_bytes(b"chunk")

        provider = DummyProvider(
            transcriptions_by_filename={
                "left_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Сәлеметсіз бе мен клиентпін",
                    raw_text="Сәлеметсіз бе мен клиентпін",
                    corrected_text="Сәлеметсіз бе мен клиентпін.",
                    model="openai/whisper-large-v3-turbo",
                    filename="left_chunk_0001.wav",
                ),
                "right_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Сәлеметсіз бе тыңдап тұрмын айта беріңіз",
                    raw_text="Сәлеметсіз бе тыңдап тұрмын айта беріңіз",
                    corrected_text="Сәлеметсіз бе тыңдап тұрмын айта беріңіз.",
                    model="openai/whisper-large-v3-turbo",
                    filename="right_chunk_0001.wav",
                ),
                "mono_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Сәлеметсіз бе",
                    raw_text="Сәлеметсіз бе",
                    corrected_text="Сәлеметсіз бе.",
                    model="openai/whisper-large-v3-turbo",
                    filename="mono_chunk_0001.wav",
                ),
                "phase_fixed_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Thank you",
                    raw_text="Thank you",
                    corrected_text="Thank you.",
                    model="openai/whisper-large-v3-turbo",
                    filename="phase_fixed_chunk_0001.wav",
                ),
                "phase_fixed_reverse_chunk_0001.wav": AudioTranscriptionResponse(
                    text="Рақмет",
                    raw_text="Рақмет",
                    corrected_text="Рақмет.",
                    model="openai/whisper-large-v3-turbo",
                    filename="phase_fixed_reverse_chunk_0001.wav",
                ),
            }
        )
        service = TTSService(settings, provider=provider)
        variants = [
            type("Variant", (), {"name": "left", "chunks": [left_chunk]})(),
            type("Variant", (), {"name": "right", "chunks": [right_chunk]})(),
            type("Variant", (), {"name": "phase_fixed", "chunks": [phase_chunk]})(),
            type("Variant", (), {"name": "phase_fixed_reverse", "chunks": [reverse_phase_chunk]})(),
            type("Variant", (), {"name": "mono", "chunks": [mono_chunk]})(),
        ]
        service._preprocessor = DummyPreprocessor(
            DummyPreparedAudio(
                chunks=[left_chunk, right_chunk, mono_chunk, phase_chunk, reverse_phase_chunk],
                duration_seconds=12.0,
                variants=variants,
                channel_count=2,
            )
        )
        service._meeting_analyzer = DummyMeetingAnalyzer(
            refined_text="Сәлеметсіз бе мен клиентпін. Сәлеметсіз бе тыңдап тұрмын айта беріңіз.",
            should_refine=True,
        )

        result = asyncio.run(
            service.transcribe_audio(
                audio_bytes=b"fake-audio",
                filename="call.wav",
                content_type="audio/wav",
                report_language="kk",
            )
        )

        assert result.text == "Сәлеметсіз бе мен клиентпін\nСәлеметсіз бе тыңдап тұрмын айта беріңіз"


def test_service_skips_refine_for_suspicious_transcript():
    settings = build_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk = Path(temp_dir) / "chunk_0001.wav"
        chunk.write_bytes(b"chunk-1")
        provider = DummyProvider(
            transcription=AudioTranscriptionResponse(
                text="Қалайсын Қалайсын Қалайсын Kisi thou var",
                raw_text="Қалайсын Қалайсын Қалайсын Kisi thou var",
                corrected_text="Қалайсын Қалайсын Қалайсын Kisi thou var",
                model="openai/whisper-large-v3-turbo",
                filename="chunk_0001.wav",
            )
        )
        service = TTSService(settings, provider=provider)
        service._preprocessor = DummyPreprocessor(DummyPreparedAudio(chunks=[chunk], duration_seconds=12.0))
        service._meeting_analyzer = DummyMeetingAnalyzer(refined_text="Hallucinated clean text.", should_refine=False)

        result = asyncio.run(
            service.transcribe_audio(
                audio_bytes=b"fake-audio",
                filename="sample.mp3",
                content_type="audio/mpeg",
                report_language="kk",
            )
        )

        assert result.corrected_text == "Қалайсын Қалайсын Қалайсын Kisi thou var."


def test_meeting_analysis_uses_fallback_summary_for_suspicious_transcript():
    settings = build_settings()
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk = Path(temp_dir) / "chunk_0001.wav"
        chunk.write_bytes(b"chunk-1")
        provider = DummyProvider(
            transcription=AudioTranscriptionResponse(
                text="bad bad bad mixed script",
                raw_text="bad bad bad mixed script",
                corrected_text="bad bad bad mixed script",
                model="openai/whisper-large-v3-turbo",
                filename="chunk_0001.wav",
            )
        )
        service = TTSService(settings, provider=provider)
        service._preprocessor = DummyPreprocessor(DummyPreparedAudio(chunks=[chunk], duration_seconds=12.0))
        service._meeting_analyzer = DummyMeetingAnalyzer(refined_text="Hallucinated clean text.", should_refine=False)
        service._analyzer = DummyAnalyzer(
            response=TextAnalysisResult(
                summary="Needs review.",
                overall_quality="acceptable",
                coherence_score=40,
                wording_score=40,
                meaning_score=20,
                pronunciation_risk_score=80,
                corrected_text="bad bad bad mixed script",
                issues=[],
                recommendations=["Review audio."],
            )
        )

        result = asyncio.run(
            service.analyze_meeting_audio(
                audio_bytes=b"fake-audio",
                filename="meeting.wav",
                content_type="audio/wav",
                report_language="kk",
            )
        )

        assert result.summary.short_summary == "Fallback summary."


def test_service_analyzes_long_text_in_chunks():
    settings = build_settings()
    settings.max_text_length = 10
    provider = DummyProvider(
        response=TTSResponse(content=b"audio", content_type="audio/mpeg", filename="speech.mp3")
    )
    service = TTSService(settings, provider=provider)
    service._analyzer = DummyAnalyzer(
        response=TextAnalysisResult(
            summary="Chunk ok.",
            overall_quality="acceptable",
            coherence_score=80,
            wording_score=82,
            meaning_score=84,
            pronunciation_risk_score=20,
            corrected_text="Chunk corrected.",
            issues=[],
            recommendations=["Tighten wording."],
        )
    )

    result = asyncio.run(service.analyze_text(TextAnalysisRequest(text="one two three four five six")))

    assert result.summary == "Сводный анализ собран по 3 текстовым фрагментам."
    assert result.overall_quality == "acceptable"
    assert result.recommendations == ["Tighten wording."]
