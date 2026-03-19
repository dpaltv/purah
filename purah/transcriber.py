import json
from pathlib import Path
from typing import Optional
import logging

from faster_whisper import WhisperModel

from . import config

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_size = model_size or config.WHISPER_MODEL
        self.device = device or config.WHISPER_DEVICE
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading faster-whisper model: {self.model_size}...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="auto",
            )
            logger.info("Model loaded successfully")
        return self._model

    def transcribe(self, video_path: Path, output_path: Optional[Path] = None) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_path is None:
            output_path = video_path.with_suffix(".transcript.json")

        logger.info(f"Transcribing {video_path} with faster-whisper ({self.model_size})...")

        segments, info = self.model.transcribe(
            str(video_path),
            language="en",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )

        segment_list = []
        for seg in segments:
            segment_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })

        result = {
            "transcription": {
                "segments": segment_list,
                "duration": info.duration or segment_list[-1]["end"] if segment_list else 0,
            }
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Transcription saved to {output_path}")
        return result

    def transcribe_to_text(self, video_path: Path) -> str:
        video_path = Path(video_path)

        logger.info(f"Transcribing {video_path} to text...")

        segments, _ = self.model.transcribe(
            str(video_path),
            beam_size=5,
        )

        return " ".join(seg.text.strip() for seg in segments)
