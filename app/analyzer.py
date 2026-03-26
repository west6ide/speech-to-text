import json
import logging
import re
import time
from typing import Any

from openai import APIConnectionError, APIError, AsyncOpenAI, AuthenticationError, RateLimitError

from app.config import Settings
from app.errors import TTSConfigurationError, TTSProviderError
from app.models import AnalysisIssue, TextAnalysisRequest, TextAnalysisResult


class TextAnalyzer:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise TTSConfigurationError("OPENAI_API_KEY is not configured")

        client_kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._settings = settings
        self._logger = logging.getLogger("app.analyzer")
        self._client = AsyncOpenAI(
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            **client_kwargs,
        )

    async def analyze(self, payload: TextAnalysisRequest) -> TextAnalysisResult:
        started_at = time.perf_counter()
        heuristic_issues = self._collect_heuristic_issues(payload.text, payload.report_language)
        self._logger.info(
            "analysis_started model=%s text_chars=%s heuristic_issues=%s report_language=%s",
            self._settings.openai_analysis_model,
            len(payload.text),
            len(heuristic_issues),
            payload.report_language,
        )
        llm_result = await self._analyze_with_llm(payload.text, heuristic_issues, payload.report_language)
        merged = self._merge_with_heuristics(llm_result, heuristic_issues, payload.text)
        self._logger.info(
            "analysis_finished model=%s duration_ms=%s overall_quality=%s issues=%s",
            self._settings.openai_analysis_model,
            round((time.perf_counter() - started_at) * 1000, 2),
            merged.overall_quality,
            len(merged.issues),
        )
        return merged

    def correct_text(self, text: str) -> str:
        return self._apply_corrections(text)

    async def _analyze_with_llm(
        self,
        text: str,
        heuristic_issues: list[AnalysisIssue],
        report_language: str,
    ) -> TextAnalysisResult:
        language_name = "Russian" if report_language == "ru" else "Kazakh"
        system_prompt = (
            "You analyze text before speech synthesis or speech transcription validation. "
            "Return only one valid JSON object and nothing else. "
            "Required fields: summary, overall_quality, coherence_score, wording_score, meaning_score, "
            "pronunciation_risk_score, corrected_text, issues, recommendations. "
            "overall_quality must be one of good, acceptable, needs_revision. "
            "issues must be an array of objects with severity, category, fragment, explanation, suggestion. "
            f"All natural-language values must be written in {language_name}. "
            "Be strict and concrete. If there are grammar, wording, spelling, morphology, or semantic issues, report them."
        )
        user_prompt = (
            "Analyze the text quality for speech-related use. "
            "Pay attention to spelling, grammar, case endings, awkward wording, clarity of meaning, and pronunciation ambiguity.\n\n"
            f"Text:\n{text}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = await self._request_analysis(messages, strict_json=True)
            parsed = self._normalize_analysis_payload(self._parse_analysis_payload(content), report_language)
            return TextAnalysisResult.model_validate(parsed)
        except AuthenticationError as exc:
            raise TTSProviderError("Authentication with analysis model failed") from exc
        except RateLimitError as exc:
            raise TTSProviderError("Analysis model rate limit exceeded") from exc
        except APIConnectionError as exc:
            raise TTSProviderError("Analysis model is unreachable") from exc
        except APIError:
            self._logger.warning("analysis_api_error_using_fallback")
            return self._build_fallback_result(text, heuristic_issues, report_language)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            self._logger.warning("analysis_invalid_json_using_recovery")
            try:
                content = await self._request_analysis(messages, strict_json=False)
                parsed = self._normalize_analysis_payload(self._parse_analysis_payload(content), report_language)
                return TextAnalysisResult.model_validate(parsed)
            except Exception:
                self._logger.warning("analysis_recovery_failed_using_fallback")
                return self._build_fallback_result(text, heuristic_issues, report_language)
        except Exception as exc:
            raise TTSProviderError("Unexpected analysis failure") from exc

    async def _request_analysis(self, messages: list[dict[str, str]], strict_json: bool) -> str:
        started_at = time.perf_counter()
        request_payload: dict[str, object] = {
            "model": self._settings.openai_analysis_model,
            "messages": messages,
            "temperature": 0.1,
        }
        if strict_json:
            request_payload["response_format"] = {"type": "json_object"}

        self._logger.info(
            "analysis_request_started model=%s strict_json=%s",
            self._settings.openai_analysis_model,
            strict_json,
        )
        response = await self._client.chat.completions.create(**request_payload)
        content = response.choices[0].message.content or ""
        self._logger.info(
            "analysis_request_finished model=%s strict_json=%s duration_ms=%s response_chars=%s",
            self._settings.openai_analysis_model,
            strict_json,
            round((time.perf_counter() - started_at) * 1000, 2),
            len(content),
        )
        return content

    @staticmethod
    def _parse_analysis_payload(content: str) -> dict[str, Any]:
        normalized = content.strip()
        if not normalized:
            raise json.JSONDecodeError("empty content", content, 0)

        candidates = [normalized]
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, flags=re.DOTALL)
        if code_block_match:
            candidates.append(code_block_match.group(1))

        json_object_match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
        if json_object_match:
            candidates.append(json_object_match.group(0))

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise json.JSONDecodeError("unable to parse analysis payload", content, 0)

    def _normalize_analysis_payload(self, payload: dict[str, Any], report_language: str) -> dict[str, Any]:
        issues = payload.get("issues")
        if not isinstance(issues, list):
            issues = []

        normalized_issues: list[dict[str, str]] = []
        for item in issues:
            if not isinstance(item, dict):
                continue
            normalized_issues.append(
                {
                    "severity": self._normalize_severity(item.get("severity")),
                    "category": self._normalize_category(item.get("category"), report_language),
                    "fragment": self._as_text(item.get("fragment"), report_language, self._fallback_fragment(report_language)),
                    "explanation": self._as_text(
                        item.get("explanation"),
                        report_language,
                        self._fallback_explanation(report_language),
                    ),
                    "suggestion": self._as_text(
                        item.get("suggestion"),
                        report_language,
                        self._fallback_suggestion(report_language),
                    ),
                }
            )

        recommendations = payload.get("recommendations")
        if not isinstance(recommendations, list):
            recommendations = []

        normalized_recommendations = [
            self._as_text(item, report_language, self._fallback_recommendation(report_language))
            for item in recommendations
            if str(item).strip()
        ]

        normalized_payload = {
            "summary": self._as_text(payload.get("summary"), report_language, self._fallback_summary(report_language)),
            "overall_quality": self._normalize_quality(payload.get("overall_quality")),
            "coherence_score": self._normalize_score(payload.get("coherence_score"), 75),
            "wording_score": self._normalize_score(payload.get("wording_score"), 75),
            "meaning_score": self._normalize_score(payload.get("meaning_score"), 75),
            "pronunciation_risk_score": self._normalize_score(payload.get("pronunciation_risk_score"), 20),
            "corrected_text": self._as_text(payload.get("corrected_text"), report_language, ""),
            "issues": normalized_issues,
            "recommendations": normalized_recommendations,
        }
        return normalized_payload

    def _build_fallback_result(
        self,
        text: str,
        heuristic_issues: list[AnalysisIssue],
        report_language: str,
    ) -> TextAnalysisResult:
        issue_penalty = min(55, len(heuristic_issues) * 11)
        pronunciation_risk = self._estimate_pronunciation_risk(text)
        coherence_score = max(30, 92 - issue_penalty)
        wording_score = max(25, 90 - issue_penalty)
        meaning_score = max(30, 88 - issue_penalty)

        if issue_penalty >= 28:
            overall_quality = "needs_revision"
        elif issue_penalty > 0:
            overall_quality = "acceptable"
        else:
            overall_quality = "good"

        recommendations = [issue.suggestion for issue in heuristic_issues]
        if not recommendations:
            recommendations = [self._fallback_recommendation(report_language)]

        self._logger.info("analysis_fallback_used heuristic_issues=%s", len(heuristic_issues))
        return TextAnalysisResult(
            summary=self._fallback_summary(report_language),
            overall_quality=overall_quality,
            coherence_score=coherence_score,
            wording_score=wording_score,
            meaning_score=meaning_score,
            pronunciation_risk_score=pronunciation_risk,
            corrected_text=self._apply_corrections(text),
            issues=list(heuristic_issues),
            recommendations=self._merge_recommendations([], recommendations),
        )

    def _merge_with_heuristics(
        self,
        llm_result: TextAnalysisResult,
        heuristic_issues: list[AnalysisIssue],
        text: str,
    ) -> TextAnalysisResult:
        issues = list(llm_result.issues)
        for issue in heuristic_issues:
            if not self._issue_exists(issues, issue):
                issues.append(issue)

        recommendations = self._merge_recommendations(
            llm_result.recommendations,
            [issue.suggestion for issue in heuristic_issues],
        )
        penalty = min(35, max(0, len(issues) - len(llm_result.issues)) * 8)
        quality = llm_result.overall_quality
        if len(issues) >= 2 and quality == "good":
            quality = "acceptable"
        if any(issue.severity == "high" for issue in issues):
            quality = "needs_revision"

        return TextAnalysisResult(
            summary=llm_result.summary,
            overall_quality=quality,
            coherence_score=max(0, min(100, llm_result.coherence_score - penalty)),
            wording_score=max(0, min(100, llm_result.wording_score - penalty)),
            meaning_score=max(0, min(100, llm_result.meaning_score - penalty // 2)),
            pronunciation_risk_score=max(
                llm_result.pronunciation_risk_score,
                self._estimate_pronunciation_risk(text),
            ),
            corrected_text=llm_result.corrected_text.strip() or self._apply_corrections(text),
            issues=issues,
            recommendations=recommendations,
        )

    def _collect_heuristic_issues(self, text: str, report_language: str) -> list[AnalysisIssue]:
        issues: list[AnalysisIssue] = []
        lower_text = text.lower()

        if re.search(r"[!?.,]{3,}", text):
            issues.append(
                AnalysisIssue(
                    severity="medium",
                    category=self._category_name("wording", report_language),
                    fragment=self._phrase("repeated_punctuation", report_language),
                    explanation=self._phrase("repeated_punctuation_explanation", report_language),
                    suggestion=self._phrase("repeated_punctuation_suggestion", report_language),
                )
            )

        if re.search(r"\b(\w+)( \1\b)+", text, flags=re.IGNORECASE):
            issues.append(
                AnalysisIssue(
                    severity="medium",
                    category=self._category_name("coherence", report_language),
                    fragment=self._phrase("repeated_words", report_language),
                    explanation=self._phrase("repeated_words_explanation", report_language),
                    suggestion=self._phrase("repeated_words_suggestion", report_language),
                )
            )

        if len(text.split()) < 3:
            issues.append(
                AnalysisIssue(
                    severity="low",
                    category=self._category_name("meaning", report_language),
                    fragment=text,
                    explanation=self._phrase("short_text_explanation", report_language),
                    suggestion=self._phrase("short_text_suggestion", report_language),
                )
            )

        known_typos = {
            "интерект": ("интерект", "интеллект"),
            "жасайтының": ("жасайтының", "жасайтынын"),
            "ссадятся": ("ссадятся", "садятся"),
            "дубовами": ("дубовами", "образами"),
        }
        for typo, (fragment, suggestion_value) in known_typos.items():
            if typo in lower_text:
                issues.append(
                    AnalysisIssue(
                        severity="high",
                        category=self._category_name("grammar", report_language),
                        fragment=fragment,
                        explanation=self._phrase("known_typo_explanation", report_language).format(fragment=fragment),
                        suggestion=self._phrase("known_typo_suggestion", report_language).format(
                            suggestion=suggestion_value
                        ),
                    )
                )

        mixed_script_words = re.findall(r"\b(?=\w*[A-Za-z])(?=\w*[А-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ])[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]+\b", text)
        for word in mixed_script_words[:3]:
            issues.append(
                AnalysisIssue(
                    severity="medium",
                    category=self._category_name("wording", report_language),
                    fragment=word,
                    explanation=self._phrase("mixed_script_explanation", report_language),
                    suggestion=self._phrase("mixed_script_suggestion", report_language),
                )
            )

        if re.search(r"\b(\d+)\b", text):
            issues.append(
                AnalysisIssue(
                    severity="low",
                    category=self._category_name("pronunciation", report_language),
                    fragment=self._phrase("numbers_fragment", report_language),
                    explanation=self._phrase("numbers_explanation", report_language),
                    suggestion=self._phrase("numbers_suggestion", report_language),
                )
            )

        return issues

    @staticmethod
    def _apply_corrections(text: str) -> str:
        corrected = f" {text.strip()} "
        replacements = {
            " интеректінің ": " интеллектінің ",
            " интерект ": " интеллект ",
            " жасайтының ": " жасайтынын ",
            " жазылып тұр ": " жазылған ",
            " ссадятся ": " садятся ",
            " дубовами ": " образами ",
            " ломб слифт ": " лонгслив ",
            " лонгслифт ": " лонгслив ",
            " лонгслик ": " лонгслив ",
        }
        for source, target in replacements.items():
            corrected = corrected.replace(source, target)

        corrected = re.sub(r"\s+", " ", corrected).strip()
        if corrected and corrected[-1] not in ".!?":
            corrected += "."
        if corrected:
            corrected = corrected[0].upper() + corrected[1:]
        return corrected

    @staticmethod
    def _normalize_quality(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"good", "acceptable", "needs_revision"}:
            return normalized
        return "acceptable"

    @staticmethod
    def _normalize_score(value: Any, default: int) -> int:
        try:
            score = int(float(value))
        except (TypeError, ValueError):
            score = default
        return max(0, min(100, score))

    @staticmethod
    def _normalize_severity(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return "medium"

    def _normalize_category(self, value: Any, report_language: str) -> str:
        mapping = {
            "grammar": self._category_name("grammar", report_language),
            "coherence": self._category_name("coherence", report_language),
            "wording": self._category_name("wording", report_language),
            "meaning": self._category_name("meaning", report_language),
            "style": self._category_name("style", report_language),
            "spelling": self._category_name("spelling", report_language),
            "semantics": self._category_name("meaning", report_language),
            "pronunciation": self._category_name("pronunciation", report_language),
        }
        normalized = str(value or "").strip().lower()
        return mapping.get(normalized, self._category_name("wording", report_language))

    @staticmethod
    def _as_text(value: Any, report_language: str, default: str) -> str:
        text = str(value or "").strip()
        return text if text else default

    @staticmethod
    def _issue_exists(existing: list[AnalysisIssue], candidate: AnalysisIssue) -> bool:
        return any(
            issue.fragment == candidate.fragment and issue.category == candidate.category
            for issue in existing
        )

    @staticmethod
    def _merge_recommendations(existing: list[str], extra: list[str]) -> list[str]:
        merged: list[str] = []
        for item in existing + extra:
            normalized = item.strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged

    @staticmethod
    def _estimate_pronunciation_risk(text: str) -> int:
        latin = len(re.findall(r"[A-Za-z]", text))
        digits = len(re.findall(r"\d", text))
        special = len(re.findall(r"[/@#&+=:_-]", text))
        mixed_script = len(re.findall(r"(?=.*[A-Za-z])(?=.*[А-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]).+", text))
        risk = min(100, latin * 2 + digits * 3 + special * 4 + mixed_script * 8)
        return max(5, risk)

    def _fallback_summary(self, report_language: str) -> str:
        if report_language == "ru":
            return "Использован резервный анализ. Отчет собран из детерминированных правил, потому что модель не вернула надежный структурированный результат."
        return "Сенімді құрылымдалған нәтиже қайтпағандықтан, есеп резервтік ережелер негізінде жасалды."

    def _fallback_recommendation(self, report_language: str) -> str:
        if report_language == "ru":
            return "Явных критических ошибок не найдено, но текст стоит проверить вручную на орфографию и естественность формулировок."
        return "Өрескел қате табылмады, бірақ мәтінді орфография мен сөйлем табиғилығы тұрғысынан қолмен қайта тексерген дұрыс."

    def _fallback_fragment(self, report_language: str) -> str:
        return "Фрагмент текста" if report_language == "ru" else "Мәтін үзіндісі"

    def _fallback_explanation(self, report_language: str) -> str:
        if report_language == "ru":
            return "Фрагмент требует дополнительной ручной проверки."
        return "Бұл үзіндіні қосымша қолмен тексеру керек."

    def _fallback_suggestion(self, report_language: str) -> str:
        if report_language == "ru":
            return "Уточните формулировку и проверьте орфографию."
        return "Сөйлемді нақтылап, орфографиясын тексеріңіз."

    def _category_name(self, key: str, report_language: str) -> str:
        categories = {
            "ru": {
                "grammar": "грамматика",
                "coherence": "связность",
                "wording": "формулировка",
                "meaning": "смысл",
                "style": "стиль",
                "spelling": "орфография",
                "pronunciation": "произношение",
            },
            "kk": {
                "grammar": "грамматика",
                "coherence": "байланыс",
                "wording": "тұжырым",
                "meaning": "мағына",
                "style": "стиль",
                "spelling": "орфография",
                "pronunciation": "айтылым",
            },
        }
        return categories[report_language][key]

    def _phrase(self, key: str, report_language: str) -> str:
        phrases = {
            "ru": {
                "repeated_punctuation": "Повторяющаяся пунктуация",
                "repeated_punctuation_explanation": "Серия одинаковых знаков препинания делает речь менее естественной и может ломать интонацию.",
                "repeated_punctuation_suggestion": "Оставьте один уместный знак препинания вместо серии символов.",
                "repeated_words": "Повторяющиеся слова подряд",
                "repeated_words_explanation": "Повтор одинаковых слов подряд часто звучит как ошибка формулировки или дефект исходного текста.",
                "repeated_words_suggestion": "Уберите дубль или перестройте фразу.",
                "short_text_explanation": "Текст слишком короткий для уверенной смысловой оценки.",
                "short_text_suggestion": "Добавьте больше контекста, если нужен надежный анализ.",
                "known_typo_explanation": "Во фрагменте «{fragment}» вероятна орфографическая или грамматическая ошибка.",
                "known_typo_suggestion": "Проверьте форму слова. Вероятный корректный вариант: «{suggestion}».",
                "mixed_script_explanation": "В одном слове смешаны разные алфавиты, это часто приводит к ошибкам распознавания и произношения.",
                "mixed_script_suggestion": "Приведите слово к одному алфавиту и проверьте написание.",
                "numbers_fragment": "Числа в тексте",
                "numbers_explanation": "Числа могут читаться неоднозначно без контекста.",
                "numbers_suggestion": "Если важна естественная озвучка, распишите числа словами.",
            },
            "kk": {
                "repeated_punctuation": "Қайталанған тыныс белгілері",
                "repeated_punctuation_explanation": "Бірдей тыныс белгілерінің тізбегі сөйлеу ырғағын бұзып, интонацияны табиғи емес етеді.",
                "repeated_punctuation_suggestion": "Бірнеше белгінің орнына бір ғана орынды тыныс белгісін қалдырыңыз.",
                "repeated_words": "Қатар қайталанған сөздер",
                "repeated_words_explanation": "Бір сөздің қатар қайталануы сөйлем табиғилығын бұзады және мәтіндегі қате болуы мүмкін.",
                "repeated_words_suggestion": "Қайталанған сөзді алып тастаңыз немесе сөйлемді қайта құрыңыз.",
                "short_text_explanation": "Мәтін мағынаны сенімді бағалау үшін тым қысқа.",
                "short_text_suggestion": "Сенімді талдау керек болса, көбірек контекст қосыңыз.",
                "known_typo_explanation": "«{fragment}» бөлігінде орфографиялық не грамматикалық қате болуы мүмкін.",
                "known_typo_suggestion": "Сөз тұлғасын тексеріңіз. Дұрысы шамамен: «{suggestion}».",
                "mixed_script_explanation": "Бір сөздің ішінде әртүрлі әліпбилер араласқан, бұл тануға және айтылымға кедергі келтіреді.",
                "mixed_script_suggestion": "Сөзді бір әліпбиге келтіріп, жазылуын тексеріңіз.",
                "numbers_fragment": "Мәтіндегі сандар",
                "numbers_explanation": "Сандар контекст болмаса, әртүрлі оқылуы мүмкін.",
                "numbers_suggestion": "Табиғи оқылым маңызды болса, сандарды сөзбен жазыңыз.",
            },
        }
        return phrases[report_language][key]
