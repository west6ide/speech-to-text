from fastapi.testclient import TestClient

from app.main import app, get_tts_service
from app.models import (
    AdvancedAudioAnalysisResponse,
    AdvancedTranscriptionResult,
    AudioProfile,
    MeetingSummary,
    TextAnalysisResult,
    TranscriptionCandidate,
)


class AdvancedStubService:
    async def analyze_audio_advanced(self, audio_bytes, filename, content_type=None, model=None, report_language="ru"):
        return AdvancedAudioAnalysisResponse(
            audio_profile=AudioProfile(
                duration_seconds=12.0,
                channel_count=2,
                variant_count=6,
                chunk_count=6,
                content_type=content_type,
                detected_call_recording=True,
                recommended_model=model or "openai/whisper-large-v3",
            ),
            transcription=AdvancedTranscriptionResult(
                text="hello world",
                corrected_text="Hello world.",
                model=model or "openai/whisper-large-v3",
                variant="phone_cleaned",
                confidence=0.91,
                suspicious=False,
                suspicious_spans=[],
            ),
            candidates=[
                TranscriptionCandidate(
                    model=model or "openai/whisper-large-v3",
                    variant="phone_cleaned",
                    text="hello world",
                    score=82.0,
                    suspicious=False,
                )
            ],
            analysis=TextAnalysisResult(
                summary="РњУ™С‚С–РЅ С‚ТЇСЃС–РЅС–РєС‚С–.",
                overall_quality="good",
                coherence_score=90,
                wording_score=90,
                meaning_score=92,
                pronunciation_risk_score=12,
                corrected_text="РўТЇР·РµС‚С–Р»РіРµРЅ РјУ™С‚С–РЅ.",
                issues=[],
                recommendations=["Р•Р»РµСѓР»С– РјУ™СЃРµР»Рµ С‚Р°Р±С‹Р»Т“Р°РЅ Р¶РѕТ›."],
            ),
            summary=MeetingSummary(
                short_summary="РљРѕСЂРѕС‚РєР°СЏ СЃРІРѕРґРєР°.",
                detailed_summary="РџРѕРґСЂРѕР±РЅР°СЏ СЃРІРѕРґРєР° РІСЃС‚СЂРµС‡Рё.",
                topics=["РћР±СЃСѓР¶РґРµРЅРёРµ"],
                decisions=[],
                action_items=[],
                open_questions=[],
                risks=[],
            ),
        )


def test_advanced_audio_analyze_endpoint():
    app.dependency_overrides[get_tts_service] = lambda: AdvancedStubService()
    client = TestClient(app)

    response = client.post(
        "/v2/audio/analyze",
        data={"report_language": "kk", "model": '"openai/whisper-large-v3"'},
        files={"file": ("call.wav", b"fake-audio", "audio/wav")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["audio_profile"]["detected_call_recording"] is True
    assert payload["transcription"]["variant"] == "phone_cleaned"
    assert payload["candidates"][0]["model"] == "openai/whisper-large-v3"
