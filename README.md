# English Stuttering-Augmentation Pipeline (LLM branch)

This reproduces the **LLM-based** half of the paper's dataset-generation pipeline
(Section 3.1.3, second strategy + Section 3.1.4), adapted from Indonesian to English:

1. **Label sampling** (`sampler.py`) — for each fluent sentence, sample 1–3 target
   disfluency categories (`repetition`, `prolongation`, `interjection`), weighted
   50/30/20 to mirror the paper's rule-based algorithm's per-word probabilities.
2. **Text disfluency injection** (`disfluency_generator.py`) — a batched
   GPT-4o-mini call, using an English version of the paper's JSON-in/JSON-out
   prompt, that rewrites each sentence to contain the assigned disfluency
   type(s) without changing meaning or word order.
3. **TTS synthesis** (`tts_generator.py`) — `gpt-4o-mini-tts`, with voice
   sampled uniformly at random and speed sampled uniformly from `[0.75, 1.25]`,
   exactly as described in the paper. The stuttered pass adds the instruction
   `"Speak with a stutter, in a natural English accent."`; the clean pass has
   no special instruction.
4. **Orchestration** (`pipeline.py`) — ties it together and writes:
   - `outputs/text/disfluent_dataset.json` — original/disfluent text pairs + labels
   - `outputs/audio_clean/*.mp3`, `outputs/audio_stutter/*.mp3` — synthesized audio
   - `outputs/manifest.csv` — one row per sentence with paths, labels, voice, speed

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Usage

Put one fluent sentence per line in a `.txt` file (or a `.csv` with a `text`
column) — `data/sentences.txt` has 10 example sentences to get you started.

**Check the text stage first, without spending on audio:**

```bash
python pipeline.py --input data/sentences.txt --dry-run-text
```

Inspect `outputs/text/disfluent_dataset.json` — confirm the disfluencies look
right and `original_text`/`disfluency_type` weren't drifted by the model.

**Run the full pipeline (text + audio):**

```bash
python pipeline.py --input data/sentences.txt --limit 20
```

Drop `--limit` to process the whole file. `--seed` controls both the label
sampling and the voice/speed sampling, for reproducibility.

## Notes / things to tune for your setup

- **Accent instruction**: `TTS_STUTTER_INSTRUCTION` in `config.py` currently
  says "natural English accent" (generic). If you want e.g. a specific
  regional accent (American, British, Indian English, etc.), just edit that
  string — it's passed straight to `gpt-4o-mini-tts` as free-text instructions.
- **Voices**: `TTS_VOICES` lists the full current OpenAI TTS voice set. If
  OpenAI adds/removes voices, update that list.
- **Batch size / retries**: `TEXT_BATCH_SIZE` (10) and `TEXT_MAX_RETRIES` (3)
  in `config.py` control the LLM stage; the generator validates that
  `original_text` wasn't altered and retries the whole batch if the model's
  JSON doesn't parse or drifts, similar to the "short reminder" the paper
  mentions adding to keep `teks_asli`/`jenis_gagap` stable.
- **Cost**: each sentence triggers 1 LLM call (batched with ~9 others) + 2 TTS
  calls (clean + stuttered). Test on a small `--limit` before scaling up.
- **I could not execute this against the real OpenAI API from this sandbox**
  (the sandbox network is restricted to a small domain allowlist and does not
  include `api.openai.com`), so I verified the code compiles and the label
  sampler behaves correctly, but you should smoke-test the actual API calls
  on your own machine before scaling up.

## Extending

- To also compare against the paper's deterministic/rule-based algorithm,
  you'd add a sibling module implementing the per-word probability logic
  described in Section 3.1.3 (0.8 global modification probability; 50/30/20
  split across repetition/prolongation/interjection) — not included here
  since you said you're focusing on the LLM branch.
- If you want to fine-tune an English Whisper checkpoint the way the paper
  fine-tunes `cahya/whisper-small-id`, the `manifest.csv` this pipeline
  produces already has the (audio_path, transcript) pairs you'd need for
  a Hugging Face `datasets`/`transformers` fine-tuning script.
