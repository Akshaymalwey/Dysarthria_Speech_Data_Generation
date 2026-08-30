"""
Text-to-speech synthesis, mirroring the paper's stage 3.1.4:

  - model: gpt-4o-mini-tts
  - speaker sampled uniformly at random from all available voices
  - speed sampled uniformly from [0.75, 1.25]
  - a short instruction string nudges *acoustic* delivery only, since the
    disfluency markers (repetitions/prolongations/interjections) are already
    baked into the text by the LLM stage
  - both clean and stuttered text get corresponding audio, for a direct
    fluent-vs-disfluent comparison
"""

import os
import random
from typing import Optional

from openai import OpenAI

from config import (
    TTS_MODEL,
    TTS_VOICES,
    TTS_SPEED_RANGE,
    TTS_STUTTER_INSTRUCTION,
    TTS_CLEAN_INSTRUCTION,
    AUDIO_FORMAT,
)


class TTSGenerator:
    def __init__(self, api_key: str = None, model: str = TTS_MODEL, seed: int = 42):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.rng = random.Random(seed)

    def _sample_voice(self) -> str:
        return self.rng.choice(TTS_VOICES)

    def _sample_speed(self) -> float:
        lo, hi = TTS_SPEED_RANGE
        return round(self.rng.uniform(lo, hi), 3)

    def synthesize(
        self,
        text: str,
        output_path: str,
        stutter: bool,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> dict:
        """
        Render `text` to an audio file at `output_path`.
        Returns metadata about the render (voice, speed, instruction used).
        """
        voice = voice or self._sample_voice()
        speed = speed if speed is not None else self._sample_speed()
        instructions = TTS_STUTTER_INSTRUCTION if stutter else TTS_CLEAN_INSTRUCTION

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        kwargs = dict(
            model=self.model,
            voice=voice,
            input=text,
            speed=speed,
            response_format=AUDIO_FORMAT,
        )
        if instructions:
            kwargs["instructions"] = instructions

        with self.client.audio.speech.with_streaming_response.create(**kwargs) as response:
            response.stream_to_file(output_path)

        return {
            "path": output_path,
            "voice": voice,
            "speed": speed,
            "stutter": stutter,
            "instructions": instructions,
        }
