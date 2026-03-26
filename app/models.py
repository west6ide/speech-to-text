from typing import Literal

from pydantic import BaseModel, Field, field_validator


SUPPORTED_AUDIO_FORMATS = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg"}
SUPPORTED_INPUT_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/webm",
    "audio/mp4",
    "audio/m4a",
}


SUPPORTED_REPORT_LANGUAGES = {"ru", "kk"}


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to convert to speech")
    voice: str | None = Field(default=None, description="Voice identifier")
    model: str | None = Field(default=None, description="TTS model name")
    format: str = Field(default="mp3", description="Audio output format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("voice must not be blank")
        return normalized

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(f"format must be one of: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}")
        return normalized


class TTSResponse(BaseModel):
    content: bytes
    content_type: str
    filename: str


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")
    report_language: Literal["ru", "kk"] = Field(
        default="ru",
        description="Language for the analysis report: ru or kk",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized


class AnalysisIssue(BaseModel):
    severity: str = Field(..., description="low, medium or high")
    category: str = Field(..., description="grammar, coherence, wording, meaning, style")
    fragment: str = Field(..., description="Problematic fragment")
    explanation: str = Field(..., description="Why this fragment may be problematic")
    suggestion: str = Field(..., description="How to improve it")


class TextAnalysisResult(BaseModel):
    summary: str
    overall_quality: str = Field(..., description="good, acceptable or needs_revision")
    coherence_score: int = Field(..., ge=0, le=100)
    wording_score: int = Field(..., ge=0, le=100)
    meaning_score: int = Field(..., ge=0, le=100)
    pronunciation_risk_score: int = Field(..., ge=0, le=100)
    corrected_text: str = Field(..., description="Suggested corrected version of the input text")
    issues: list[AnalysisIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class TTSReportResponse(BaseModel):
    analysis: TextAnalysisResult
    audio_base64: str
    content_type: str
    filename: str


class AudioTranscriptionResponse(BaseModel):
    text: str
    raw_text: str
    corrected_text: str
    model: str
    filename: str
    duration_seconds: float | None = None
    chunk_count: int = 1


class AudioAnalysisReportResponse(BaseModel):
    transcription: AudioTranscriptionResponse
    analysis: TextAnalysisResult


class MeetingActionItem(BaseModel):
    owner: str
    task: str
    deadline: str | None = None


class MeetingSummary(BaseModel):
    short_summary: str
    detailed_summary: str
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[MeetingActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class MeetingAnalysisResponse(BaseModel):
    transcription: AudioTranscriptionResponse
    analysis: TextAnalysisResult
    summary: MeetingSummary
