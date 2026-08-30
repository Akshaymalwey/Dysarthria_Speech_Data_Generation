"""
End-to-end pipeline:

  1. Load fluent English sentences (one per line, or a CSV column).
  2. Sample disfluency category/categories per sentence.
  3. Batch-call the LLM to inject stutter-like disfluencies into the text.
  4. Synthesize both the clean and the disfluent text with gpt-4o-mini-tts,
     matching the paper's speaker/speed sampling.
  5. Write a manifest.csv tying everything together (paths, labels, voice,
     speed) plus a JSON dump of the text-only dataset.

Usage:
    export OPENAI_API_KEY=sk-...
    python pipeline.py --input data/sentences.txt --limit 20

Run with --dry-run-text to only do step 1-3 (no TTS calls / no audio cost)
so you can sanity-check the disfluent text before spending on audio.
"""

import argparse
import csv
import json
import os

from config import (
    TEXT_OUTPUT_DIR,
    AUDIO_CLEAN_DIR,
    AUDIO_STUTTER_DIR,
    MANIFEST_PATH,
    AUDIO_FORMAT,
)
from sampler import assign_labels
from disfluency_generator import DisfluencyGenerator
from tts_generator import TTSGenerator


def load_sentences(path: str, limit: int = None):
    sentences = []
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col = "text" if "text" in reader.fieldnames else reader.fieldnames[0]
            for row in reader:
                s = row[col].strip()
                if s:
                    sentences.append(s)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    sentences.append(s)
    if limit:
        sentences = sentences[:limit]
    return sentences


def run(args):
    os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
    sentences = load_sentences(args.input, args.limit)
    print(f"Loaded {len(sentences)} sentences from {args.input}")

    labeled = assign_labels(sentences, seed=args.seed)

    gen = DisfluencyGenerator()
    dataset = gen.generate(labeled)

    text_json_path = os.path.join(TEXT_OUTPUT_DIR, "disfluent_dataset.json")
    with open(text_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Wrote disfluent text dataset -> {text_json_path}")

    if args.dry_run_text:
        print("--dry-run-text set: skipping TTS stage.")
        return

    os.makedirs(AUDIO_CLEAN_DIR, exist_ok=True)
    os.makedirs(AUDIO_STUTTER_DIR, exist_ok=True)

    tts = TTSGenerator(seed=args.seed)
    manifest_rows = []

    for idx, item in enumerate(dataset):
        uid = f"{idx:05d}"
        clean_path = os.path.join(AUDIO_CLEAN_DIR, f"{uid}.{AUDIO_FORMAT}")
        stutter_path = os.path.join(AUDIO_STUTTER_DIR, f"{uid}.{AUDIO_FORMAT}")

        print(f"[{uid}] synthesizing clean audio...")
        clean_meta = tts.synthesize(item["original_text"], clean_path, stutter=False)

        print(f"[{uid}] synthesizing stuttered audio...")
        stutter_meta = tts.synthesize(item["disfluent_text"], stutter_path, stutter=True)

        manifest_rows.append({
            "id": uid,
            "original_text": item["original_text"],
            "disfluent_text": item["disfluent_text"],
            "disfluency_type": "|".join(item["disfluency_type"]),
            "clean_audio_path": clean_meta["path"],
            "clean_voice": clean_meta["voice"],
            "clean_speed": clean_meta["speed"],
            "stutter_audio_path": stutter_meta["path"],
            "stutter_voice": stutter_meta["voice"],
            "stutter_speed": stutter_meta["speed"],
        })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote manifest -> {MANIFEST_PATH}")
    print(f"Clean audio -> {AUDIO_CLEAN_DIR}/")
    print(f"Stuttered audio -> {AUDIO_STUTTER_DIR}/")


def parse_args():
    p = argparse.ArgumentParser(description="English stuttering-augmentation pipeline")
    p.add_argument("--input", required=True, help="Path to .txt (one sentence/line) or .csv (column 'text')")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N sentences")
    p.add_argument("--seed", type=int, default=42, help="Seed for label sampling / voice+speed sampling")
    p.add_argument("--dry-run-text", action="store_true",
                   help="Only run the LLM disfluency-injection stage; skip TTS (no audio cost)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
