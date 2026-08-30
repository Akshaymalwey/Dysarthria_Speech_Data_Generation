"""
Configuration for the English stuttering-augmentation pipeline.

This mirrors the design choices described in the paper
"Stuttering-Aware Automatic Speech Recognition for Indonesian Language"
(LLM-based text disfluency generation stage + gpt-4o-mini-tts synthesis stage),
adapted for English source transcripts.
"""

import os

# ---------------------------------------------------------------------------
# OpenAI models
# ---------------------------------------------------------------------------
# Text-disfluency-injection model. The paper used Mistral Small 3 in an
# earlier pass and GPT-4o-mini in a later, batch-oriented pass. We follow
# the later (GPT-4o-mini, batched, JSON-in/JSON-out) design here.
TEXT_MODEL = os.environ.get("STUTTER_TEXT_MODEL", "gpt-4o-mini")

# TTS model used in the paper for stutter-speech synthesis.
TTS_MODEL = os.environ.get("STUTTER_TTS_MODEL", "gpt-4o-mini-tts")

# ---------------------------------------------------------------------------
# Disfluency taxonomy (same three categories as the paper)
# ---------------------------------------------------------------------------
DISFLUENCY_TYPES = ["repetition", "prolongation", "interjection"]

# ---------------------------------------------------------------------------
# Sampling of disfluency labels per sentence
# ---------------------------------------------------------------------------
# The paper samples "the target disfluency category or categories" per
# sentence beforehand. We allow 1-3 categories per sentence with configurable
# weights (repetition is the most common in real stuttering, so it's weighted
# highest, matching the paper's rule-based algorithm: 50% repetition,
# 30% prolongation, 20% interjection for a selected word).
DISFLUENCY_TYPE_WEIGHTS = {
    "repetition": 0.5,
    "prolongation": 0.3,
    "interjection": 0.2,
}
MIN_TYPES_PER_SENTENCE = 1
MAX_TYPES_PER_SENTENCE = 3

# ---------------------------------------------------------------------------
# LLM batching
# ---------------------------------------------------------------------------
TEXT_BATCH_SIZE = 10          # sentences per LLM call
TEXT_MAX_RETRIES = 3
TEXT_TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# TTS settings (mirrors the paper: speaker sampled uniformly at random from
# all available voices, speed sampled uniformly from [0.75, 1.25])
# ---------------------------------------------------------------------------
TTS_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo",
    "fable", "onyx", "nova", "sage", "shimmer", "verse",
]
TTS_SPEED_RANGE = (0.75, 1.25)

# Instruction passed to the TTS model for the *stuttered* audio, mirroring
# "Bicara dengan gagap (stutter) dalam aksen indonesia" but for English.
# Change the accent phrase if you want a specific English accent instead of
# a generic/neutral one.
TTS_STUTTER_INSTRUCTION = "Speak with a stutter, in a natural English accent."

# The paper generates clean (non-stuttered) audio too, for direct comparison.
# No special instruction is needed for the clean pass.
TTS_CLEAN_INSTRUCTION = None

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
TEXT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "text")
AUDIO_CLEAN_DIR = os.path.join(OUTPUT_DIR, "audio_clean")
AUDIO_STUTTER_DIR = os.path.join(OUTPUT_DIR, "audio_stutter")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")

AUDIO_FORMAT = "mp3"  # "mp3", "wav", "opus", "flac", "aac", "pcm" all supported by the TTS endpoint
