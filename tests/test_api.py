from fastapi.testclient import TestClient

from app.main import app, get_tts_service
from app.models import AudioAnalysisReportResponse, AudioTranscriptionResponse, MeetingAnalysisResponse, MeetingSummary, TTSReportResponse, TTSResponse, TextAnalysisResult


class StubService:
    async def synthesize(self, payload):
        return TTSResponse(content=b"audio", content_type="audio/mpeg", filename="speech.mp3")

    async def analyze_text(self, payload):
        report_language = getattr(payload, "report_language", "ru")
        summary = "Текст понятный." if report_language == "ru" else "Мәтін түсінікті."
        return TextAnalysisResult(
            summary=summary,
            overall_quality="good",
            coherence_score=90,
            wording_score=90,
            meaning_score=92,
            pronunciation_risk_score=12,
            corrected_text="Исправленный текст." if report_language == "ru" else "Түзетілген мәтін.",
            issues=[],
            recommendations=["Серьезных проблем не найдено." if report_language == "ru" else "Елеулі мәселе табылған жоқ."],
        )

    async def synthesize_with_report(self, payload):
        return TTSReportResponse(
            analysis=await self.analyze_text(payload),
            audio_base64="YXVkaW8=",
            content_type="audio/mpeg",
            filename="speech.mp3",
        )

    async def transcribe_audio(self, audio_bytes, filename, content_type=None, model=None, report_language="ru"):
        return AudioTranscriptionResponse(
            text="hello world",
            raw_text="hello world",
            corrected_text="Hello world.",
            model=model or "openai/whisper-large-v3-turbo",
            filename=filename,
        )

    async def transcribe_and_analyze_audio(self, audio_bytes, filename, content_type=None, model=None, report_language="ru"):
        return AudioAnalysisReportResponse(
            transcription=await self.transcribe_audio(audio_bytes, filename, content_type, model),
            analysis=await self.analyze_text(type("Payload", (), {"report_language": report_language})()),
        )

    async def analyze_meeting_audio(self, audio_bytes, filename, content_type=None, model=None, report_language="ru"):
        return MeetingAnalysisResponse(
            transcription=await self.transcribe_audio(audio_bytes, filename, content_type, model),
            analysis=await self.analyze_text(type("Payload", (), {"report_language": report_language})()),
            summary=MeetingSummary(
                short_summary="Короткая сводка.",
                detailed_summary="Подробная сводка встречи.",
                topics=["Обсуждение"],
                decisions=["Продолжить работу"],
                action_items=[],
                open_questions=[],
                risks=[],
            ),
        )


def test_healthcheck():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_text_to_speech_endpoint():
    app.dependency_overrides[get_tts_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post(
        "/v1/text-to-speech",
        json={"text": "hello", "format": "mp3", "speed": 1.0},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.content == b"audio"
    assert response.headers["content-type"] == "audio/mpeg"


def test_text_analysis_endpoint():
    app.dependency_overrides[get_tts_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post("/v1/text-analysis", json={"text": "hello world"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["overall_quality"] == "good"
    assert response.json()["coherence_score"] == 90


def test_text_analysis_endpoint_kazakh():
    app.dependency_overrides[get_tts_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post("/v1/text-analysis", json={"text": "hello world", "report_language": "kk"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["summary"] == "Мәтін түсінікті."


def test_text_to_speech_report_endpoint():
    app.dependency_overrides[get_tts_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post(
        "/v1/text-to-speech/report",
        json={"text": "hello world", "format": "mp3", "speed": 1.0},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["audio_base64"] == "YXVkaW8="
    assert response.json()["analysis"]["overall_quality"] == "good"


def test_speech_to_text_endpoint():
    app.dependency_overrides[get_tts_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post(
        "/v1/speech-to-text",
        files={"file": ("sample.mp3", b"fake-audio", "audio/mpeg")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["text"] == "hello world"
    assert response.json()["corrected_text"] == "Hello world."


def test_speech_to_text_report_endpoint():
    app.dependency_overrides[get_tts_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post(
        "/v1/speech-to-text/report",
        data={"report_language": "ru"},
        files={"file": ("sample.mp3", b"fake-audio", "audio/mpeg")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["transcription"]["text"] == "hello world"
    assert response.json()["analysis"]["overall_quality"] == "good"


def test_meeting_analyze_endpoint():
    app.dependency_overrides[get_tts_service] = lambda: StubService()
    client = TestClient(app)

    response = client.post(
        "/v1/meeting/analyze",
        data={"report_language": "ru"},
        files={"file": ("meeting.wav", b"fake-audio", "audio/wav")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["transcription"]["corrected_text"] == "Hello world."
    assert response.json()["summary"]["short_summary"] == "Короткая сводка."
