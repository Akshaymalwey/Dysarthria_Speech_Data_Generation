"""
Text-to-speech synthesis using Microsoft's SpeechT5 (local, free), in place
of gpt-4o-mini-tts.

Mirrors the paper's stage 3.1.4 design intent as closely as a
non-instruction-following local TTS model allows:

  - speaker sampled uniformly at random from a pool of CMU Arctic speakers
    (free x-vector embeddings), giving voice variety analogous to the
    paper's "sampled uniformly from all available voices"
  - speed sampled uniformly from [0.75, 1.25], applied as a post-hoc
    time-stretch on the generated waveform (SpeechT5 has no native speed
    control, unlike gpt-4o-mini-tts)
  - both clean and stuttered text get corresponding audio, for a direct
    fluent-vs-disfluent comparison

Model loading (SpeechT5 + vocoder + speaker embeddings) happens once in
__init__ and is reused across all synthesize() calls -- load one
TTSGenerator instance per pipeline run, not per sentence.
"""

import os
import random
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

from config import (
    TTS_MODEL,
    TTS_VOCODER,
    SPEAKER_EMBEDDINGS_DATASET,
    SPEAKER_EMBEDDINGS_SPLIT,
    SPEAKER_IDS,
    NUM_SPEAKERS,
    SAMPLE_RATE,
    TTS_DEVICE,
    TTS_SPEED_RANGE,
    MAX_TTS_TEXT_CHARS,
)


def _resolve_device(device_setting: str) -> torch.device:
    if device_setting == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_setting == "cpu":
        return torch.device("cpu")
    # "auto"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_speaker_embeddings() -> List[Tuple[str, torch.Tensor]]:
    """
    Load a pool of CMU Arctic x-vector speaker embeddings (free, via
    HuggingFace datasets). Tries the requested speaker IDs first, falls
    back to the first N available embeddings if lookup fails, and finally
    falls back to a direct zip download if the HF dataset script path
    fails (mirrors the reference open-source implementation's approach).
    """
    from datasets import load_dataset

    available = []
    try:
        ds = load_dataset(SPEAKER_EMBEDDINGS_DATASET, split=SPEAKER_EMBEDDINGS_SPLIT)

        keep_cols = [c for c in ds.column_names if c in {"xvector", "speaker_id", "speaker", "id", "name"}]
        if keep_cols:
            remove_cols = [c for c in ds.column_names if c not in keep_cols]
            if remove_cols:
                ds = ds.remove_columns(remove_cols)

        for i, row in enumerate(ds):
            if len(available) >= NUM_SPEAKERS:
                break
            if "xvector" not in row:
                continue
            speaker_id = row.get("speaker_id") or row.get("speaker") or f"spk_{i}"
            available.append((str(speaker_id), torch.tensor(row["xvector"]).float()))

    except Exception as e:  # noqa: BLE001
        print(f"[tts_generator] HF dataset load failed ({e}); trying direct zip download...")
        import io
        import urllib.request
        import zipfile

        url = f"https://huggingface.co/datasets/{SPEAKER_EMBEDDINGS_DATASET}/resolve/main/spkrec-xvect.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                speaker_files = {}
                for name in z.namelist():
                    if name.endswith(".npy"):
                        spk = name.split("/")[-1].split("-")[0]
                        speaker_files.setdefault(spk, name)
                for spk, name in list(speaker_files.items())[:NUM_SPEAKERS]:
                    with z.open(name) as f:
                        arr = np.load(f)
                        available.append((spk, torch.tensor(arr).float()))

    if not available:
        raise RuntimeError(
            f"Could not load any speaker embeddings from {SPEAKER_EMBEDDINGS_DATASET}"
        )
    return available


class TTSGenerator:
    def __init__(self, seed: int = 42, device: str = TTS_DEVICE):
        self.device = _resolve_device(device)
        self.rng = random.Random(seed)

        print(f"[tts_generator] loading SpeechT5 on {self.device} ...")
        self.processor = SpeechT5Processor.from_pretrained(TTS_MODEL)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL).to(self.device)
        self.model.eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained(TTS_VOCODER).to(self.device)
        self.vocoder.eval()

        print("[tts_generator] loading speaker embeddings...")
        speakers = _load_speaker_embeddings()
        self.speaker_map = dict(speakers)
        self.speaker_ids = [s for s, _ in speakers]
        print(f"[tts_generator] loaded {len(self.speaker_ids)} speakers: {self.speaker_ids}")

    def _sample_voice(self) -> str:
        return self.rng.choice(self.speaker_ids)

    def _sample_speed(self) -> float:
        lo, hi = TTS_SPEED_RANGE
        return round(self.rng.uniform(lo, hi), 3)

    def _synthesize_raw(self, text: str, speaker_embedding: torch.Tensor) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        speaker_embedding = speaker_embedding.unsqueeze(0).to(self.device)

        with torch.no_grad():
            speech = self.model.generate_speech(input_ids, speaker_embedding, vocoder=self.vocoder)

        return speech.cpu().numpy()

    def synthesize(
        self,
        text: str,
        output_path: str,
        stutter: bool,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> dict:
        """
        Render `text` to a wav file at `output_path`.
        `stutter` is kept for interface/manifest compatibility with the
        OpenAI-based generator; SpeechT5 has no delivery-style instruction,
        so the disfluent audio's "stutter" comes entirely from the
        repetition/prolongation/interjection markers already present in
        the text.
        Returns metadata about the render (voice, speed used).
        """
        if len(text) > MAX_TTS_TEXT_CHARS:
            raise ValueError(
                f"Text too long for TTS ({len(text)} > {MAX_TTS_TEXT_CHARS} chars): {text[:60]}..."
            )

        voice = voice or self._sample_voice()
        speed = speed if speed is not None else self._sample_speed()
        speaker_embedding = self.speaker_map[voice]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        audio = self._synthesize_raw(text, speaker_embedding)

        # Apply speed variation via time-stretch, matching the paper's
        # [0.75, 1.25] uniform sampling (SpeechT5 has no native speed knob).
        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)

        sf.write(output_path, audio, SAMPLE_RATE)

        return {
            "path": output_path,
            "voice": voice,
            "speed": speed,
            "stutter": stutter,
        }
