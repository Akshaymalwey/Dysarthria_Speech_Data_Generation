# English Stuttering-Augmentation Pipeline (LLM Branch, Free Stack)

This reproduces the **LLM-based** half of the paper's dataset-generation
pipeline (Section 3.1.3, second strategy + Section 3.1.4), adapted from
Indonesian to English, using a **fully free tool stack**:

- **Text disfluency injection**: Groq's free-tier API (OpenAI-compatible),
  in place of GPT-4o-mini.
- **TTS**: Microsoft **SpeechT5**, running locally via HuggingFace
  `transformers` (free, no API key, no per-request cost), in place of
  `gpt-4o-mini-tts`. Speaker variety comes from free CMU Arctic x-vector
  embeddings.

## Pipeline

1. **Label sampling** (`sampler.py`) — for each fluent sentence, sample 1–3
   target disfluency categories (`repetition`, `prolongation`,
   `interjection`), weighted 50/30/20 to mirror the paper's rule-based
   algorithm's per-word category split.
2. **Text disfluency injection** (`disfluency_generator.py`) — a batched
   Groq chat-completion call, using an English version of the paper's
   JSON-in/JSON-out prompt, that rewrites each sentence to contain the
   assigned disfluency type(s) without changing meaning or word order.
3. **TTS synthesis** (`tts_generator.py`) — local SpeechT5 + HiFi-GAN
   vocoder. Speaker sampled uniformly at random from a pool of CMU Arctic
   voices; speed sampled uniformly from `[0.75, 1.25]` and applied as a
   post-hoc time-stretch (SpeechT5 has no native speed control, unlike
   `gpt-4o-mini-tts`). Both clean and disfluent text are synthesized for
   direct comparison.
4. **Orchestration** (`pipeline.py`) — ties it together and writes:
   - `outputs/text/disfluent_dataset.json` — original/disfluent text pairs + labels
   - `outputs/audio_clean/*.wav`, `outputs/audio_stutter/*.wav` — synthesized audio
   - `outputs/manifest.csv` — one row per sentence with paths, labels, voice, speed

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key at https://console.groq.com/keys, then:

```bash
export GROQ_API_KEY="..."
```

No key or signup is needed for the TTS step — SpeechT5 and the CMU Arctic
speaker embeddings download automatically from HuggingFace on first run and
are cached locally afterward (a few hundred MB total).

## Usage

Put one fluent sentence per line in a `.txt` file (or a `.csv` with a `text`
column) — `data/sentences.txt` has 10 example sentences to get you started.

**Check the text stage first, without downloading/running SpeechT5:**

```bash
python pipeline.py --input data/sentences.txt --dry-run-text
```

Inspect `outputs/text/disfluent_dataset.json` — confirm the disfluencies
look right and `original_text`/`disfluency_type` weren't drifted by the
model.

**Run the full pipeline (text + audio):**

```bash
python pipeline.py --input data/sentences.txt --limit 20
```

Drop `--limit` to process the whole file. `--seed` controls both the label
sampling and the voice/speed sampling, for reproducibility. The first call
that reaches the TTS stage will download SpeechT5 + vocoder + speaker
embeddings; subsequent runs reuse the local cache.

## Notes / things to tune for your setup

- **Groq model**: `TEXT_MODEL` in `config.py` defaults to
  `llama-3.3-70b-versatile` (best instruction-following on the free tier).
  If you hit rate limits, try a smaller/faster model like
  `llama-3.1-8b-instant` via `STUTTER_TEXT_MODEL` env var.
- **Voices**: `SPEAKER_IDS` in `config.py` lists 8 common CMU Arctic
  speakers (mixed accents/genders). If a listed ID isn't found in the
  embeddings dataset, the loader falls back to the first N available
  embeddings automatically.
- **Speed variation**: implemented via `librosa.effects.time_stretch` since
  SpeechT5 doesn't take a speed parameter the way `gpt-4o-mini-tts` does.
- **No delivery instruction**: `gpt-4o-mini-tts` could be told "speak with a
  stutter" to nudge acoustic delivery. SpeechT5 has no equivalent — the
  audible stutter comes entirely from the repetition/prolongation/filler
  markers already present in the text from step 2. This should still work,
  since that's how the paper describes the instruction's role anyway
  (a nudge on top of text that's already disfluent, not the source of it).
- **Batch size / retries**: `TEXT_BATCH_SIZE` (10) and `TEXT_MAX_RETRIES`
  (3) in `config.py` control the LLM stage; the generator validates that
  `original_text` wasn't altered and retries the whole batch if the JSON
  doesn't parse or drifts.
- **Text length cap**: `MAX_TTS_TEXT_CHARS` (250) guards against SpeechT5
  instability on very long inputs, mirroring the reference implementation's
  own cap.
- **Compute**: SpeechT5 runs on CPU (slow but workable for a BTP-scale
  dataset) or CUDA if available (`STUTTER_TTS_DEVICE=cuda`). `TTS_DEVICE`
  defaults to `"auto"`.
- **I could not execute this against the real Groq/SpeechT5 stack from this
  sandbox** (restricted network + no Groq key here), so I verified: all
  modules import and byte-compile cleanly, the label sampler behaves
  correctly, and the Groq client raises the expected clear error when no
  API key is set. Smoke-test the actual Groq calls and a first SpeechT5
  download on your own machine before scaling up.

## Extending

- To also compare against the paper's deterministic/rule-based algorithm,
  the reference repo above already has a clean, English, YAML-configurable
  implementation (`augment.py` + `configs/augment_initial.yaml`) you could
  adapt as a sibling branch for direct comparison against this LLM branch.
- If you want to fine-tune an English Whisper checkpoint the way the paper
  fine-tunes `cahya/whisper-small-id`, the `manifest.csv` this pipeline
  produces already has the (audio_path, transcript) pairs you'd need for
  a Hugging Face `datasets`/`transformers` fine-tuning script.
