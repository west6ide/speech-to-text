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
class PreparedAudioVariant:
    name: str
    chunks: list[Path]
    normalized_path: Path


@dataclass
class PreparedAudio:
    temp_dir: tempfile.TemporaryDirectory[str]
    variants: list[PreparedAudioVariant]
    chunks: list[Path]
    duration_seconds: float
    channel_count: int

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

        metadata = self._probe_audio_metadata(input_path)
        variants = self._prepare_variants(input_path, workdir, metadata["channels"])
        chunk_paths = [chunk for variant in variants for chunk in variant.chunks]
        self._logger.info(
            "audio_prepared filename=%s duration_seconds=%s channel_count=%s variant_count=%s chunk_count=%s duration_ms=%s",
            filename,
            round(metadata["duration_seconds"], 2),
            metadata["channels"],
            len(variants),
            len(chunk_paths),
            round((time.perf_counter() - started_at) * 1000, 2),
        )
        return PreparedAudio(
            temp_dir=temp_dir,
            variants=variants,
            chunks=chunk_paths,
            duration_seconds=metadata["duration_seconds"],
            channel_count=metadata["channels"],
        )

    def _ensure_ffmpeg_available(self) -> None:
        if shutil.which(self._settings.ffmpeg_path) is None:
            raise TTSProviderError(f"ffmpeg executable not found: {self._settings.ffmpeg_path}")
        if shutil.which(self._settings.ffprobe_path) is None:
            raise TTSProviderError(f"ffprobe executable not found: {self._settings.ffprobe_path}")

    def _probe_audio_metadata(self, input_path: Path) -> dict[str, float | int]:
        result = subprocess.run(
            [
                self._settings.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=channels",
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
            streams = payload.get("streams") or []
            channels = max(int(stream.get("channels") or 0) for stream in streams) if streams else 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TTSProviderError("Unable to determine audio metadata") from exc
        if duration <= 0:
            raise TTSProviderError("Audio duration must be greater than zero")
        return {"duration_seconds": duration, "channels": max(1, channels)}

    def _prepare_variants(self, input_path: Path, workdir: Path, channel_count: int) -> list[PreparedAudioVariant]:
        variants: list[PreparedAudioVariant] = []
        if channel_count >= 2:
            for name, channel_expr in (("left", "pan=mono|c0=c0"), ("right", "pan=mono|c0=c1")):
                normalized_path = workdir / f"{name}.wav"
                self._normalize_audio(input_path, normalized_path, audio_filter=channel_expr)
                variants.append(
                    PreparedAudioVariant(
                        name=name,
                        chunks=self._split_audio(normalized_path),
                        normalized_path=normalized_path,
                    )
                )

            for name, channel_expr in (
                ("phase_fixed", "pan=mono|c0=c0-c1"),
                ("phase_fixed_reverse", "pan=mono|c0=c1-c0"),
                ("phone_cleaned", "pan=mono|c0=c0-c1,highpass=f=120,lowpass=f=3800,afftdn,dynaudnorm"),
                ("phone_cleaned_reverse", "pan=mono|c0=c1-c0,highpass=f=120,lowpass=f=3800,afftdn,dynaudnorm"),
            ):
                normalized_path = workdir / f"{name}.wav"
                self._normalize_audio(input_path, normalized_path, audio_filter=channel_expr)
                variants.append(
                    PreparedAudioVariant(
                        name=name,
                        chunks=self._split_audio(normalized_path),
                        normalized_path=normalized_path,
                    )
                )

        normalized_path = workdir / "mono.wav"
        self._normalize_audio(input_path, normalized_path, audio_filter="pan=mono|c0=0.5*c0+0.5*c1" if channel_count >= 2 else None)
        variants.append(
            PreparedAudioVariant(
                name="mono",
                chunks=self._split_audio(normalized_path),
                normalized_path=normalized_path,
            )
        )
        cleaned_path = workdir / "mono_cleaned.wav"
        cleaned_filter = (
            "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=120,lowpass=f=3800,afftdn,dynaudnorm"
            if channel_count >= 2
            else "highpass=f=120,lowpass=f=3800,afftdn,dynaudnorm"
        )
        self._normalize_audio(input_path, cleaned_path, audio_filter=cleaned_filter)
        variants.append(
            PreparedAudioVariant(
                name="mono_cleaned",
                chunks=self._split_audio(cleaned_path),
                normalized_path=cleaned_path,
            )
        )
        return variants

    def _normalize_audio(self, input_path: Path, output_path: Path, audio_filter: str | None = None) -> None:
        command = [
            self._settings.ffmpeg_path,
            "-y",
            "-i",
            str(input_path),
        ]
        if audio_filter:
            command.extend(["-af", audio_filter])
        command.extend(
            [
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                str(output_path),
            ]
        )
        result = subprocess.run(
            command,
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

        expected_chunks = max(
            1,
            math.ceil(
                float(self._probe_audio_metadata(normalized_path)["duration_seconds"])
                / self._settings.stt_chunk_duration_seconds
            ),
        )
        self._logger.info(
            "audio_split_finished normalized=%s chunk_count=%s expected_chunk_count=%s",
            normalized_path.name,
            len(chunks),
            expected_chunks,
        )
        return chunks
