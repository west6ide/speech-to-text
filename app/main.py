import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response

from app.config import Settings, get_settings
from app.errors import TTSConfigurationError
from app.logging_config import configure_logging
from app.models import (
    AudioAnalysisReportResponse,
    AudioTranscriptionResponse,
    MeetingAnalysisResponse,
    TTSReportResponse,
    TTSRequest,
    TextAnalysisRequest,
    TextAnalysisResult,
)
from app.service import TTSService


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logging.getLogger("app").info(
        "application_started base_url=%s tts_model=%s stt_model=%s analysis_model=%s timeout_seconds=%s",
        settings.openai_base_url or "-",
        settings.openai_tts_model,
        settings.openai_stt_model,
        settings.openai_analysis_model,
        settings.openai_timeout_seconds,
    )
    yield


app = FastAPI(title="Text-to-Speech Service", version="0.1.0", lifespan=lifespan)
logger = logging.getLogger("app.http")


def _clean_form_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1].strip()
    return normalized


def get_tts_service(settings: Settings = Depends(get_settings)) -> TTSService:
    try:
        return TTSService(settings)
    except TTSConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    started_at = time.perf_counter()
    logger.info("request_started id=%s method=%s path=%s", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed id=%s path=%s", request_id, request.url.path)
        raise

    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "request_finished id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def healthcheck() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/v1/text-to-speech")
async def text_to_speech(
    payload: TTSRequest,
    service: TTSService = Depends(get_tts_service),
) -> Response:
    result = await service.synthesize(payload)
    headers = {"Content-Disposition": f'attachment; filename="{result.filename}"'}
    return Response(content=result.content, media_type=result.content_type, headers=headers)


@app.post("/v1/text-analysis", response_model=TextAnalysisResult)
async def text_analysis(
    payload: TextAnalysisRequest,
    service: TTSService = Depends(get_tts_service),
) -> TextAnalysisResult:
    return await service.analyze_text(payload)


@app.post("/v1/text-to-speech/report", response_model=TTSReportResponse)
async def text_to_speech_report(
    payload: TTSRequest,
    service: TTSService = Depends(get_tts_service),
) -> TTSReportResponse:
    return await service.synthesize_with_report(payload)


@app.post("/v1/speech-to-text", response_model=AudioTranscriptionResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    report_language: str = Form(default="ru"),
    service: TTSService = Depends(get_tts_service),
) -> AudioTranscriptionResponse:
    audio_bytes = await file.read()
    return await service.transcribe_audio(
        audio_bytes=audio_bytes,
        filename=file.filename or "audio",
        content_type=file.content_type,
        model=_clean_form_value(model),
        report_language=_clean_form_value(report_language) or "ru",
    )


@app.post("/v1/speech-to-text/report", response_model=AudioAnalysisReportResponse)
async def speech_to_text_report(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    report_language: str = Form(default="ru"),
    service: TTSService = Depends(get_tts_service),
) -> AudioAnalysisReportResponse:
    audio_bytes = await file.read()
    return await service.transcribe_and_analyze_audio(
        audio_bytes=audio_bytes,
        filename=file.filename or "audio",
        content_type=file.content_type,
        model=_clean_form_value(model),
        report_language=_clean_form_value(report_language) or "ru",
    )


@app.post("/v1/meeting/analyze", response_model=MeetingAnalysisResponse)
async def meeting_analyze(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    report_language: str = Form(default="ru"),
    service: TTSService = Depends(get_tts_service),
) -> MeetingAnalysisResponse:
    audio_bytes = await file.read()
    return await service.analyze_meeting_audio(
        audio_bytes=audio_bytes,
        filename=file.filename or "audio",
        content_type=file.content_type,
        model=_clean_form_value(model),
        report_language=_clean_form_value(report_language) or "ru",
    )
