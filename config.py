"""
Configuration for the English stuttering-augmentation pipeline (free stack).

Text disfluency injection : Groq free-tier LLM API (OpenAI-compatible)
TTS                        : Microsoft SpeechT5 (local, free, via HuggingFace transformers)

This mirrors the design of the paper
"Stuttering-Aware Automatic Speech Recognition for Indonesian Language",
adapted for English source transcripts, using free/open tooling in place
of GPT-4o-mini and gpt-4o-mini-tts.
"""

import os

# ---------------------------------------------------------------------------
# Text disfluency injection (Groq)
# ---------------------------------------------------------------------------
# Groq's free/developer tier currently serves these as production models:
# openai/gpt-oss-120b (higher quality) and openai/gpt-oss-20b (faster).
# Note: llama-3.3-70b-versatile / llama-3.1-8b-instant have moved to
# Groq's Enterprise tier and are no longer reachable on a free-tier key
# (they now 404 with "model_not_found"). openai/gpt-oss-120b is a solid
# default for the JSON-in/JSON-out instruction-following this pipeline needs.
TEXT_MODEL = os.environ.get("STUTTER_TEXT_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# Disfluency taxonomy (same three categories as the paper)
# ---------------------------------------------------------------------------
DISFLUENCY_TYPES = ["repetition", "prolongation", "interjection"]

# ---------------------------------------------------------------------------
# Sampling of disfluency labels per sentence
# ---------------------------------------------------------------------------
# 1-3 categories per sentence, weighted to mirror the paper's rule-based
# algorithm's per-word split (50% repetition, 30% prolongation, 20%
# interjection).
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
# TTS (SpeechT5, local + free)
# ---------------------------------------------------------------------------
TTS_MODEL = os.environ.get("STUTTER_TTS_MODEL", "microsoft/speecht5_tts")
TTS_VOCODER = os.environ.get("STUTTER_TTS_VOCODER", "microsoft/speecht5_hifigan")

# CMU Arctic x-vector speaker embeddings, same free source the reference
# open-source implementation (rohitkhatri-314/Statistical-Modelling-of-
# Disfluent-Speech) uses for speaker variety.
SPEAKER_EMBEDDINGS_DATASET = "Matthijs/cmu-arctic-xvectors"
SPEAKER_EMBEDDINGS_SPLIT = "validation"
# Common CMU Arctic speaker IDs (mix of US/Scottish/Canadian English accents,
# male and female). If exact ID lookup fails, the loader falls back to the
# first N available embeddings.
SPEAKER_IDS = ["slt", "bdl", "clb", "rms", "awb", "jmk", "ksp", "lnh"]
NUM_SPEAKERS = 8

SAMPLE_RATE = 16000  # SpeechT5's native output rate
TTS_DEVICE = os.environ.get("STUTTER_TTS_DEVICE", "auto")  # "auto" | "cpu" | "cuda"

# The paper samples playback speed uniformly from [0.75, 1.25]. SpeechT5 has
# no native speed control, so we apply this as a post-hoc time-stretch
# (librosa.effects.time_stretch) on the generated waveform to match the
# paper's variability in delivery speed.
TTS_SPEED_RANGE = (0.75, 1.25)

MAX_TTS_TEXT_CHARS = 250  # guards against SpeechT5 instability on very long inputs

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
TEXT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "text")
AUDIO_CLEAN_DIR = os.path.join(OUTPUT_DIR, "audio_clean")
AUDIO_STUTTER_DIR = os.path.join(OUTPUT_DIR, "audio_stutter")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")

AUDIO_FORMAT = "wav"  # soundfile writes wav/flac natively without extra codecs