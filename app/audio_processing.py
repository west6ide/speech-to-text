import json
import logging
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.errors import TTSProviderError


@dataclass
class PreparedAudio:
    temp_dir: tempfile.TemporaryDirectory
    chunks: list[Path]
    duration_seconds: float
    normalized_path: Path

    def cleanup(self) -> None:
        self.temp_dir.cleanup()


class AudioPreprocessor:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._logger = logging.getLogger("app.audio")

    def prepare(self, audio_bytes: bytes, filename: str) -> PreparedAudio:
        started_at = time.perf_counter()
        self._ensure_ffmpeg_available()

        temp_dir = tempfile.TemporaryDirectory(prefix="stt-")
        workdir = Path(temp_dir.name)
        input_path = workdir / filename
        input_path.write_bytes(audio_bytes)

        duration_seconds = self._probe_duration(input_path)
        normalized_path = workdir / "normalized.wav"
        self._normalize_audio(input_path, normalized_path)

        chunk_paths = self._split_audio(normalized_path)
        self._logger.info(
            "audio_prepared filename=%s duration_seconds=%s chunk_count=%s duration_ms=%s",
            filename,
            round(duration_seconds, 2),
            len(chunk_paths),
            round((time.perf_counter() - started_at) * 1000, 2),
        )
        return PreparedAudio(
            temp_dir=temp_dir,
            chunks=chunk_paths,
            duration_seconds=duration_seconds,
            normalized_path=normalized_path,
        )

    def _ensure_ffmpeg_available(self) -> None:
        if shutil.which(self._settings.ffmpeg_path) is None:
            raise TTSProviderError(f"ffmpeg executable not found: {self._settings.ffmpeg_path}")
        if shutil.which(self._settings.ffprobe_path) is None:
            raise TTSProviderError(f"ffprobe executable not found: {self._settings.ffprobe_path}")

    def _probe_duration(self, input_path: Path) -> float:
        result = subprocess.run(
            [
                self._settings.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise TTSProviderError(f"ffprobe failed: {result.stderr.strip() or result.stdout.strip()}")

        try:
            payload = json.loads(result.stdout)
            duration = float(payload["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TTSProviderError("Unable to determine audio duration") from exc
        if duration <= 0:
            raise TTSProviderError("Audio duration must be greater than zero")
        return duration

    def _normalize_audio(self, input_path: Path, output_path: Path) -> None:
        result = subprocess.run(
            [
                self._settings.ffmpeg_path,
                "-y",
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not output_path.exists():
            raise TTSProviderError(f"ffmpeg normalize failed: {result.stderr.strip() or result.stdout.strip()}")

    def _split_audio(self, normalized_path: Path) -> list[Path]:
        chunk_dir = normalized_path.parent / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_pattern = chunk_dir / "chunk_%04d.wav"

        result = subprocess.run(
            [
                self._settings.ffmpeg_path,
                "-y",
                "-i",
                str(normalized_path),
                "-f",
                "segment",
                "-segment_time",
                str(self._settings.stt_chunk_duration_seconds),
                "-c",
                "copy",
                str(chunk_pattern),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise TTSProviderError(f"ffmpeg split failed: {result.stderr.strip() or result.stdout.strip()}")

        chunks = sorted(chunk_dir.glob("chunk_*.wav"))
        if not chunks:
            raise TTSProviderError("Audio split produced no chunks")

        expected_chunks = max(1, math.ceil(self._probe_duration(normalized_path) / self._settings.stt_chunk_duration_seconds))
        self._logger.info(
            "audio_split_finished normalized=%s chunk_count=%s expected_chunk_count=%s",
            normalized_path.name,
            len(chunks),
            expected_chunks,
        )
        return chunks
