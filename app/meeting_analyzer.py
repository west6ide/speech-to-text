import json
import logging
import re
import time
from typing import Any

from openai import APIConnectionError, APIError, AsyncOpenAI, AuthenticationError, RateLimitError

from app.config import Settings
from app.errors import TTSConfigurationError
from app.models import MeetingActionItem, MeetingSummary


class MeetingAnalyzer:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise TTSConfigurationError("OPENAI_API_KEY is not configured")

        client_kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._settings = settings
        self._logger = logging.getLogger("app.meeting")
        self._client = AsyncOpenAI(
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            **client_kwargs,
        )

    def should_refine_transcript(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False

        words = re.findall(r"\w+", normalized, flags=re.UNICODE)
        if not words:
            return False

        unique_words = {word.casefold() for word in words}
        if len(words) >= 8 and (len(unique_words) / len(words)) < 0.35:
            return False

        latin_words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", normalized)
        cyrillic_words = re.findall(r"\b[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі][А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі'-]*\b", normalized)
        dominant_script_words = max(len(latin_words), len(cyrillic_words))
        minority_script_words = min(len(latin_words), len(cyrillic_words))
        if dominant_script_words >= 3 and minority_script_words >= 2:
            return False

        return True

    async def refine_transcript(self, text: str, report_language: str) -> str:
        if not text.strip():
            return text

        language_name = "Russian" if report_language == "ru" else "Kazakh"
        messages = [
            {
                "role": "system",
                "content": (
                    f"You clean speech-to-text transcripts and rewrite them into natural {language_name}. "
                    "Preserve meaning. Fix obvious recognition mistakes, punctuation, grammar, and formatting. "
                    "Return only the corrected text without explanations."
                ),
            },
            {"role": "user", "content": text},
        ]

        started_at = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.openai_analysis_model,
                messages=messages,
                temperature=0.1,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                self._logger.info(
                    "meeting_refine_finished duration_ms=%s raw_chars=%s clean_chars=%s",
                    round((time.perf_counter() - started_at) * 1000, 2),
                    len(text),
                    len(content),
                )
                return content
        except (AuthenticationError, RateLimitError, APIConnectionError, APIError):
            self._logger.warning("meeting_refine_failed_using_raw_text")
        except Exception:
            self._logger.warning("meeting_refine_unexpected_failure_using_raw_text")
        return text

    def fallback_summary(self, text: str, report_language: str) -> MeetingSummary:
        return self._fallback_summary(text, report_language)

    async def summarize_meeting(self, text: str, report_language: str) -> MeetingSummary:
        try:
            if len(text) > self._settings.max_text_length:
                return await self._summarize_long_meeting(text, report_language)

            payload = await self._request_summary_payload(text, report_language)
            return self._normalize_summary(payload, report_language)
        except Exception:
            self._logger.warning("meeting_summary_fallback_used")
            return self._fallback_summary(text, report_language)

    async def _summarize_long_meeting(self, text: str, report_language: str) -> MeetingSummary:
        chunks = self._split_text(text)
        self._logger.info(
            "meeting_summary_long_started chunk_count=%s text_chars=%s",
            len(chunks),
            len(text),
        )

        chunk_payloads: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            try:
                payload = await self._request_summary_payload(chunk, report_language)
                chunk_payloads.append(payload)
                self._logger.info(
                    "meeting_summary_long_chunk_finished index=%s text_chars=%s",
                    index,
                    len(chunk),
                )
            except Exception:
                self._logger.warning(
                    "meeting_summary_long_chunk_failed index=%s text_chars=%s",
                    index,
                    len(chunk),
                )
                chunk_payloads.append(self._fallback_summary(chunk, report_language).model_dump())

        try:
            merged_payload = await self._merge_chunk_summaries(chunk_payloads, report_language)
            return self._normalize_summary(merged_payload, report_language)
        except Exception:
            self._logger.warning("meeting_summary_long_merge_failed")
            return self._fallback_merge_chunk_summaries(chunk_payloads, report_language)

    async def _request_summary_payload(self, text: str, report_language: str) -> dict[str, Any]:
        language_name = "Russian" if report_language == "ru" else "Kazakh"
        messages = [
            {
                "role": "system",
                "content": (
                    "You analyze transcripts and return only one valid JSON object with fields: "
                    "short_summary, detailed_summary, topics, decisions, action_items, open_questions, risks. "
                    "Capture the actual substance of the text, not just the opening lines. "
                    "short_summary must be 2-4 informative sentences. "
                    "detailed_summary must be a compact but informative paragraph. "
                    "topics, decisions, open_questions, risks must be arrays of short strings. "
                    "action_items must be an array of objects with owner, task, deadline. "
                    "If the input is not a business meeting, still summarize the actual content correctly. "
                    f"All natural-language values must be written in {language_name}."
                ),
            },
            {"role": "user", "content": text},
        ]
        response = await self._client.chat.completions.create(
            model=self._settings.openai_analysis_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = (response.choices[0].message.content or "").strip()
        return self._parse_json(content)

    async def _merge_chunk_summaries(
        self,
        chunk_payloads: list[dict[str, Any]],
        report_language: str,
    ) -> dict[str, Any]:
        language_name = "Russian" if report_language == "ru" else "Kazakh"
        messages = [
            {
                "role": "system",
                "content": (
                    "You merge chunk-level summaries into one final JSON object with fields: "
                    "short_summary, detailed_summary, topics, decisions, action_items, open_questions, risks. "
                    "Do not repeat the opening words of the source unless they matter. "
                    "Focus on the main events, developments, outcomes, and unresolved points. "
                    "If the source is a story or audiobook, summarize the plot and major turning points instead of meeting decisions. "
                    f"All natural-language values must be written in {language_name}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"chunks": chunk_payloads}, ensure_ascii=False),
            },
        ]
        response = await self._client.chat.completions.create(
            model=self._settings.openai_analysis_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = (response.choices[0].message.content or "").strip()
        return self._parse_json(content)

    def _normalize_summary(self, payload: dict[str, Any], report_language: str) -> MeetingSummary:
        action_items_raw = payload.get("action_items")
        if not isinstance(action_items_raw, list):
            action_items_raw = []

        action_items = []
        for item in action_items_raw:
            if not isinstance(item, dict):
                continue
            action_items.append(
                MeetingActionItem(
                    owner=str(item.get("owner") or self._unknown_owner(report_language)).strip(),
                    task=str(item.get("task") or self._no_task_text(report_language)).strip(),
                    deadline=(str(item.get("deadline")).strip() if item.get("deadline") else None),
                )
            )

        return MeetingSummary(
            short_summary=str(payload.get("short_summary") or self._fallback_short_summary(report_language)).strip(),
            detailed_summary=str(payload.get("detailed_summary") or self._fallback_detailed_summary(report_language)).strip(),
            topics=self._normalize_str_list(payload.get("topics")),
            decisions=self._normalize_str_list(payload.get("decisions")),
            action_items=action_items,
            open_questions=self._normalize_str_list(payload.get("open_questions")),
            risks=self._normalize_str_list(payload.get("risks")),
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        if not content:
            raise json.JSONDecodeError("empty", content, 0)
        candidates = [content]
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("invalid", content, 0)

    @staticmethod
    def _normalize_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _fallback_summary(self, text: str, report_language: str) -> MeetingSummary:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        short_summary = " ".join(sentences[:3]) if sentences else (
            "Текст распознан, но итог собран по резервным правилам."
            if report_language == "ru"
            else "Мәтін танылды, бірақ қорытынды резервтік ережемен жиналды."
        )
        detailed_summary = " ".join(sentences[:8]) if sentences else short_summary
        topics = self._extract_topics(text, report_language)
        open_questions = [sentence for sentence in sentences if "?" in sentence][:5]
        return MeetingSummary(
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            topics=topics,
            decisions=[],
            action_items=[],
            open_questions=open_questions,
            risks=[],
        )

    def _extract_topics(self, text: str, report_language: str) -> list[str]:
        lower = text.lower()
        patterns = {
            "ru": [
                ("аудиокниг", "Аудиокнига"),
                ("рассказ", "Рассказ"),
                ("сюжет", "Сюжет"),
                ("герой", "Главный герой"),
                ("деревн", "Деревня"),
                ("лес", "Лес"),
                ("ритуал", "Ритуал"),
                ("жертв", "Жертвоприношение"),
                ("чудовищ", "Чудовище"),
                ("проклят", "Проклятие"),
                ("встреч", "Обсуждение встречи"),
                ("задач", "Задачи"),
                ("решен", "Решения"),
            ],
            "kk": [
                ("аудиокітап", "Аудиокітап"),
                ("әңгі", "Әңгіме"),
                ("сюжет", "Сюжет"),
                ("ауыл", "Ауыл"),
                ("орман", "Орман"),
                ("құрбан", "Құрбандық"),
                ("құбыжық", "Құбыжық"),
                ("қарғыс", "Қарғыс"),
                ("кездес", "Кездесу"),
                ("тапсыр", "Тапсырмалар"),
                ("шешім", "Шешімдер"),
            ],
        }
        topics = []
        for needle, label in patterns[report_language]:
            if needle in lower and label not in topics:
                topics.append(label)
        return topics[:8]

    def _split_text(self, text: str) -> list[str]:
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

    def _fallback_merge_chunk_summaries(
        self,
        chunk_payloads: list[dict[str, Any]],
        report_language: str,
    ) -> MeetingSummary:
        short_parts: list[str] = []
        detailed_parts: list[str] = []
        topics: list[str] = []
        decisions: list[str] = []
        open_questions: list[str] = []
        risks: list[str] = []
        action_items: list[MeetingActionItem] = []

        for payload in chunk_payloads:
            normalized = self._normalize_summary(payload, report_language)
            if normalized.short_summary:
                short_parts.append(normalized.short_summary)
            if normalized.detailed_summary:
                detailed_parts.append(normalized.detailed_summary)
            for value in normalized.topics:
                if value not in topics:
                    topics.append(value)
            for value in normalized.decisions:
                if value not in decisions:
                    decisions.append(value)
            for value in normalized.open_questions:
                if value not in open_questions:
                    open_questions.append(value)
            for value in normalized.risks:
                if value not in risks:
                    risks.append(value)
            for item in normalized.action_items:
                duplicate = any(
                    existing.owner == item.owner
                    and existing.task == item.task
                    and existing.deadline == item.deadline
                    for existing in action_items
                )
                if not duplicate:
                    action_items.append(item)

        return MeetingSummary(
            short_summary=" ".join(short_parts[:2]).strip() or self._fallback_short_summary(report_language),
            detailed_summary=" ".join(detailed_parts[:4]).strip() or self._fallback_detailed_summary(report_language),
            topics=topics[:8],
            decisions=decisions[:10],
            action_items=action_items[:10],
            open_questions=open_questions[:10],
            risks=risks[:10],
        )

    def _unknown_owner(self, report_language: str) -> str:
        return "Не указан" if report_language == "ru" else "Көрсетілмеген"

    def _no_task_text(self, report_language: str) -> str:
        return "Задача не уточнена" if report_language == "ru" else "Тапсырма нақтыланбаған"

    def _fallback_short_summary(self, report_language: str) -> str:
        return "Краткая сводка содержания." if report_language == "ru" else "Мазмұнның қысқаша қорытындысы."

    def _fallback_detailed_summary(self, report_language: str) -> str:
        return "Подробная сводка содержания." if report_language == "ru" else "Мазмұнның толық қорытындысы."
