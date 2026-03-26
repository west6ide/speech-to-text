import asyncio
import base64
import logging
import re
import time

from fastapi import HTTPException, status

from app.analyzer import TextAnalyzer
from app.audio_processing import AudioPreprocessor
from app.config import Settings
from app.errors import TTSConfigurationError, TTSProviderError
from app.meeting_analyzer import MeetingAnalyzer
from app.models import (
    AudioAnalysisReportResponse,
    MeetingAnalysisResponse,
    AudioTranscriptionResponse,
    SUPPORTED_INPUT_AUDIO_TYPES,
    TTSReportResponse,
    TTSRequest,
    TTSResponse,
    TextAnalysisRequest,
    TextAnalysisResult,
)
from app.providers.base import BaseTTSProvider
from app.providers.openai_provider import OpenAITTSProvider


class TTSService:
    def __init__(self, settings: Settings, provider: BaseTTSProvider | None = None):
        self._settings = settings
        self._provider = provider or self._build_provider(settings)
        self._analyzer = TextAnalyzer(settings)
        self._meeting_analyzer = MeetingAnalyzer(settings)
        self._preprocessor = AudioPreprocessor(settings)
        self._logger = logging.getLogger("app.service")

    @staticmethod
    def _build_provider(settings: Settings) -> BaseTTSProvider:
        if settings.tts_provider == "openai":
            return OpenAITTSProvider(settings)
        raise TTSConfigurationError(f"Unsupported TTS provider: {settings.tts_provider}")

    async def synthesize(self, payload: TTSRequest) -> TTSResponse:
        started_at = time.perf_counter()
        if len(payload.text) > self._settings.max_text_length:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"text exceeds MAX_TEXT_LENGTH={self._settings.max_text_length}",
            )

        try:
            result = await self._provider.synthesize(payload)
            self._logger.info("service_tts_finished duration_ms=%s", round((time.perf_counter() - started_at) * 1000, 2))
            return result
        except TTSConfigurationError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except TTSProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    async def analyze_text(self, payload: TextAnalysisRequest) -> TextAnalysisResult:
        started_at = time.perf_counter()
        try:
            if len(payload.text) > self._settings.max_text_length:
                result = await self._analyze_long_text_with_language(payload.text, payload.report_language)
            else:
                result = await self._analyzer.analyze(payload)
            self._logger.info("service_analysis_finished duration_ms=%s", round((time.perf_counter() - started_at) * 1000, 2))
            return result
        except TTSConfigurationError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except TTSProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    async def synthesize_with_report(self, payload: TTSRequest) -> TTSReportResponse:
        analysis = await self.analyze_text(TextAnalysisRequest(text=payload.text, report_language="ru"))
        audio = await self.synthesize(payload)
        return TTSReportResponse(
            analysis=analysis,
            audio_base64=base64.b64encode(audio.content).decode("ascii"),
            content_type=audio.content_type,
            filename=audio.filename,
        )

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None = None,
        model: str | None = None,
        report_language: str = "ru",
    ) -> AudioTranscriptionResponse:
        started_at = time.perf_counter()
        report_language = self._sanitize_report_language(report_language)
        model = self._sanitize_model(model)
        content_type = self._normalize_content_type(content_type)
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="audio file is empty",
            )
        if content_type and content_type not in SUPPORTED_INPUT_AUDIO_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unsupported audio content type: {content_type}",
            )

        prepared_audio = None
        try:
            prepared_audio = self._preprocessor.prepare(audio_bytes=audio_bytes, filename=filename)
            result = await self._transcribe_prepared_audio(
                prepared_audio=prepared_audio,
                filename=filename,
                model=model,
                report_language=report_language,
            )
            self._logger.info("service_stt_finished duration_ms=%s", round((time.perf_counter() - started_at) * 1000, 2))
            return result
        except TTSConfigurationError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except TTSProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        finally:
            if prepared_audio is not None:
                prepared_audio.cleanup()

    async def transcribe_and_analyze_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None = None,
        model: str | None = None,
        report_language: str = "ru",
    ) -> AudioAnalysisReportResponse:
        started_at = time.perf_counter()
        transcription = await self.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            model=model,
            report_language=report_language,
        )
        analysis = await self.analyze_text(
            TextAnalysisRequest(text=transcription.text, report_language=report_language)
        )
        self._logger.info(
            "service_stt_report_finished duration_ms=%s transcript_chars=%s",
            round((time.perf_counter() - started_at) * 1000, 2),
            len(transcription.text),
        )
        return AudioAnalysisReportResponse(transcription=transcription, analysis=analysis)

    async def analyze_meeting_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None = None,
        model: str | None = None,
        report_language: str = "ru",
    ) -> MeetingAnalysisResponse:
        started_at = time.perf_counter()
        transcription = await self.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            model=model,
            report_language=report_language,
        )
        analysis = await self.analyze_text(
            TextAnalysisRequest(text=transcription.corrected_text, report_language=report_language)
        )
        if self._meeting_analyzer.should_refine_transcript(transcription.raw_text):
            summary = await self._meeting_analyzer.summarize_meeting(
                text=transcription.corrected_text,
                report_language=report_language,
            )
        else:
            summary = self._meeting_analyzer.fallback_summary(
                text=transcription.raw_text,
                report_language=report_language,
            )
        self._logger.info(
            "service_meeting_analysis_finished duration_ms=%s transcript_chars=%s",
            round((time.perf_counter() - started_at) * 1000, 2),
            len(transcription.corrected_text),
        )
        return MeetingAnalysisResponse(
            transcription=transcription,
            analysis=analysis,
            summary=summary,
        )

    async def _analyze_long_text(self, text: str) -> TextAnalysisResult:
        return await self._analyze_long_text_with_language(text=text, report_language="ru")

    async def _analyze_long_text_with_language(self, text: str, report_language: str) -> TextAnalysisResult:
        text_chunks = self._split_text_for_analysis(text)
        self._logger.info(
            "service_long_text_analysis_started chunk_count=%s text_chars=%s report_language=%s",
            len(text_chunks),
            len(text),
            report_language,
        )
        results: list[TextAnalysisResult] = []
        for index, chunk in enumerate(text_chunks, start=1):
            self._logger.info(
                "service_long_text_analysis_chunk_started index=%s text_chars=%s",
                index,
                len(chunk),
            )
            results.append(await self._analyzer.analyze(TextAnalysisRequest(text=chunk, report_language=report_language)))

        return self._merge_analysis_results(results, report_language)

    async def _transcribe_prepared_audio(
        self,
        prepared_audio,
        filename: str,
        model: str | None,
        report_language: str,
    ) -> AudioTranscriptionResponse:
        semaphore = asyncio.Semaphore(max(1, self._settings.stt_max_parallel_chunks))

        async def transcribe_chunk(index: int, chunk_path, variant_name: str) -> tuple[int, str]:
            async with semaphore:
                chunk_bytes = chunk_path.read_bytes()
                self._logger.info(
                    "service_stt_chunk_started variant=%s index=%s filename=%s chunk_bytes=%s",
                    variant_name,
                    index,
                    chunk_path.name,
                    len(chunk_bytes),
                )
                response = await self._provider.transcribe(
                    audio_bytes=chunk_bytes,
                    filename=chunk_path.name,
                    content_type="audio/wav",
                    model=model,
                    report_language=report_language,
                )
                self._logger.info(
                    "service_stt_chunk_finished variant=%s index=%s filename=%s transcript_chars=%s",
                    variant_name,
                    index,
                    chunk_path.name,
                    len(response.text),
                )
                return index, response.text.strip()

        async def transcribe_variant(variant) -> tuple[str, str]:
            tasks = [
                asyncio.create_task(transcribe_chunk(index, chunk_path, variant.name))
                for index, chunk_path in enumerate(variant.chunks, start=1)
            ]
            parts = await asyncio.gather(*tasks)
            ordered_text = " ".join(text for _, text in sorted(parts, key=lambda item: item[0]) if text).strip()
            return variant.name, ordered_text

        variant_results = await asyncio.gather(
            *(transcribe_variant(variant) for variant in prepared_audio.variants)
        )
        ordered_text = self._select_transcription_text(dict(variant_results))
        if not ordered_text:
            raise TTSProviderError("STT pipeline produced empty transcription")

        corrected_text = self._analyzer.correct_text(ordered_text)
        if self._meeting_analyzer.should_refine_transcript(corrected_text):
            corrected_text = await self._meeting_analyzer.refine_transcript(
                corrected_text,
                report_language=report_language,
            )

        return AudioTranscriptionResponse(
            text=ordered_text,
            raw_text=ordered_text,
            corrected_text=corrected_text,
            model=model or self._settings.openai_stt_model,
            filename=filename,
            duration_seconds=prepared_audio.duration_seconds,
            chunk_count=len(prepared_audio.chunks),
        )

    @staticmethod
    def _normalize_content_type(content_type: str | None) -> str | None:
        if content_type is None:
            return None
        normalized = content_type.strip().lower()
        return "audio/wav" if normalized == "audio/wave" else normalized

    @staticmethod
    def _sanitize_model(model: str | None) -> str | None:
        if model is None:
            return None
        normalized = model.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            normalized = normalized[1:-1].strip()
        return normalized or None

    @staticmethod
    def _sanitize_report_language(report_language: str) -> str:
        normalized = report_language.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            normalized = normalized[1:-1].strip()
        return normalized if normalized in {"ru", "kk"} else "ru"

    def _select_transcription_text(self, variant_results: dict[str, str]) -> str:
        scored = {
            name: (text, self._score_transcription_candidate(text))
            for name, text in variant_results.items()
            if text.strip()
        }
        if not scored:
            return ""

        if "left" in scored and "right" in scored:
            left_text, left_score = scored["left"]
            right_text, right_score = scored["right"]
            if (
                left_score >= 35
                and right_score >= 35
                and left_text.casefold() != right_text.casefold()
                and not self._is_mostly_duplicate(left_text, right_text)
            ):
                return f"{left_text}\n{right_text}".strip()

        best_name, (best_text, best_score) = max(scored.items(), key=lambda item: item[1][1])
        self._logger.info("service_stt_variant_selected variant=%s score=%s", best_name, round(best_score, 2))
        return best_text

    def _score_transcription_candidate(self, text: str) -> float:
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        if not words:
            return 0.0

        unique_ratio = len({word.casefold() for word in words}) / len(words)
        repeated_sequences = len(re.findall(r"\b(\w+)( \1\b)+", text, flags=re.IGNORECASE))
        latin_words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text))
        cyrillic_words = len(re.findall(r"\b[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі][А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі'-]*\b", text))
        dominant_script = max(latin_words, cyrillic_words)
        minority_script = min(latin_words, cyrillic_words)
        mixed_script_penalty = 20 if dominant_script >= 3 and minority_script >= 2 else 0
        length_bonus = min(len(words), 40)
        return round(unique_ratio * 60 + length_bonus - repeated_sequences * 20 - mixed_script_penalty, 2)

    @staticmethod
    def _is_mostly_duplicate(left_text: str, right_text: str) -> bool:
        left_words = {word.casefold() for word in re.findall(r"\w+", left_text, flags=re.UNICODE)}
        right_words = {word.casefold() for word in re.findall(r"\w+", right_text, flags=re.UNICODE)}
        if not left_words or not right_words:
            return False
        overlap = len(left_words & right_words) / max(len(left_words), len(right_words))
        return overlap > 0.8

    def _split_text_for_analysis(self, text: str) -> list[str]:
        limit = self._settings.max_text_length
        words = text.split()
        if not words:
            return [text]

        chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0
        for word in words:
            added_len = len(word) + (1 if current_words else 0)
            if current_words and current_len + added_len > limit:
                chunks.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)
            else:
                current_words.append(word)
                current_len += added_len

        if current_words:
            chunks.append(" ".join(current_words))
        return chunks

    def _merge_analysis_results(self, results: list[TextAnalysisResult], report_language: str) -> TextAnalysisResult:
        if not results:
            raise TTSProviderError("Long text analysis produced no results")

        recommendations: list[str] = []
        issues = []
        quality_rank = {"good": 0, "acceptable": 1, "needs_revision": 2}
        overall_quality = "good"

        for result in results:
            issues.extend(result.issues)
            for item in result.recommendations:
                normalized = item.strip()
                if normalized and normalized not in recommendations:
                    recommendations.append(normalized)
            if quality_rank[result.overall_quality] > quality_rank[overall_quality]:
                overall_quality = result.overall_quality

        avg = lambda values: round(sum(values) / len(values))
        summary = (
            f"Сводный анализ собран по {len(results)} текстовым фрагментам."
            if report_language == "ru"
            else f"Жиынтық талдау {len(results)} мәтін бөлігі бойынша жасалды."
        )
        fallback_recommendation = (
            "Во всех фрагментах не найдено существенных проблем."
            if report_language == "ru"
            else "Барлық бөліктер бойынша елеулі мәселе табылған жоқ."
        )
        return TextAnalysisResult(
            summary=summary,
            overall_quality=overall_quality,
            coherence_score=avg([result.coherence_score for result in results]),
            wording_score=avg([result.wording_score for result in results]),
            meaning_score=avg([result.meaning_score for result in results]),
            pronunciation_risk_score=max(result.pronunciation_risk_score for result in results),
            corrected_text=" ".join(result.corrected_text.strip() for result in results if result.corrected_text.strip()),
            issues=issues,
            recommendations=recommendations or [fallback_recommendation],
        )
