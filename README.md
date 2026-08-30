# English Stuttering-Augmentation Pipeline (LLM Branch)
### BTP Progress Summary — for advisor review

## 1. Background

This work adapts the **LLM-based text disfluency injection** strategy from
*"Stuttering-Aware Automatic Speech Recognition for Indonesian Language"*
(Muhammad et al.) to **English**. The original paper's pipeline had two
complementary text-augmentation branches feeding a shared TTS stage:

1. A **deterministic/rule-based algorithm** (per-word probability of
   modification, then type chosen 50% repetition / 30% prolongation / 20%
   interjection).
2. An **LLM-prompted rewrite** (Mistral Small 3, then GPT-4o-mini in a later
   pass) that injects more naturalistic, context-aware disfluencies.

**This project focuses on reproducing branch (2) for English**, then feeding
the output into the same TTS stage the paper used (`gpt-4o-mini-tts`) to
produce paired clean/disfluent audio for downstream ASR fine-tuning
(e.g. fine-tuning Whisper, as the paper does for Indonesian).

## 2. Pipeline Overview

```
fluent English sentences
        │
        ▼
[1] sampler.py            → sample 1–3 disfluency categories per sentence
        │                    (repetition / prolongation / interjection,
        │                     weighted 50/30/20 to mirror the paper's
        │                     rule-based algorithm's category split)
        ▼
[2] disfluency_generator.py → batched LLM call (JSON-in/JSON-out) that
        │                      rewrites each sentence to contain the
        │                      assigned disfluency type(s), preserving
        │                      meaning and word order
        ▼
[3] tts_generator.py       → synthesizes BOTH the original (clean) and
        │                     disfluent text to audio, mirroring the
        │                     paper's TTS settings: voice sampled uniformly
        │                     at random, speed sampled uniformly from
        │                     [0.75, 1.25]
        ▼
[4] pipeline.py            → orchestrates 1–3, writes:
                               outputs/text/disfluent_dataset.json
                               outputs/audio_clean/*.mp3
                               outputs/audio_stutter/*.mp3
                               outputs/manifest.csv  (paths + labels + voice/speed)
```

## 3. Design decisions mirroring the paper

| Aspect | Paper (Indonesian) | This implementation (English) |
|---|---|---|
| Disfluency taxonomy | repetition, prolongation, interjection | same three categories |
| Label sampling | sampled per-sentence beforehand | same; 1–3 categories, weighted 50/30/20 |
| LLM prompt structure | batched JSON in/out, `teks_asli`/`jenis_gagap`/`teks_gagap` | English equivalent: `original_text` / `disfluency_type` / `disfluent_text`, with the same "don't drift the input fields" guard the paper added for GPT-4o-mini |
| TTS model | `gpt-4o-mini-tts` | same |
| Voice sampling | uniform random from all available voices | same |
| Speed sampling | uniform from [0.75, 1.25] | same |
| Stutter instruction | "Bicara dengan gagap (stutter) dalam aksen indonesia" | "Speak with a stutter, in a natural English accent." |
| Clean audio | generated in parallel for direct comparison | same |

## 4. Current status

- Core pipeline is fully implemented and code-verified (compiles cleanly;
  the label sampler was unit-tested and produces valid, non-empty category
  lists).
- **Blocked on the OpenAI API side**: the text-generation step (GPT-4o-mini)
  returns `insufficient_quota` — the OpenAI account used for testing has no
  billing credits. TTS calls haven't been tested end-to-end yet for the
  same reason.
- No real English disfluent-text or audio samples have been generated yet;
  everything below the config/sampler layer is implemented but unverified
  against the live API.

## 5. Open decision — need advisor input

To keep iterating without incurring API cost during development, I'm
considering swapping the two paid components for free alternatives. This
changes the architecture somewhat, so I'd like your take on which direction
makes sense for the BTP (and whether it's worth keeping GPT-4o-mini /
`gpt-4o-mini-tts` as the "final" pipeline once we have budget, using free
tools only for development/iteration).

**Text generation (currently GPT-4o-mini):**

| Option | Cost | Requires | Notes |
|---|---|---|---|
| GPT-4o-mini (current) | Paid (pay-as-you-go) | OpenAI billing | Matches paper exactly; best instruction-following for the JSON contract |
| Local model via Ollama | Free | ~4–8GB model download, local compute | Fully offline; quality depends on model (e.g. Llama 3, Mistral); may need more prompt-engineering/retries for reliable JSON |
| Free-tier cloud API (e.g. Groq) | Free (quota-limited) | Internet + free API key | Fast, decent open models (Llama/Mixtral); rate-limited but no cost |

**TTS (currently `gpt-4o-mini-tts`):**

| Option | Cost | Requires | Notes |
|---|---|---|---|
| `gpt-4o-mini-tts` (current) | Paid | OpenAI billing | Matches paper exactly; supports a natural-language "speak with a stutter" delivery instruction on top of the disfluent text |
| edge-tts | Free | Internet, no signup | Uses Microsoft's online voices; many accents; supports speed control; does **not** take delivery instructions — disfluency comes entirely from the text markers already injected in step 2 |
| Piper | Free | Local install, no internet at runtime | Fully offline neural TTS; same instruction limitation as edge-tts |

Losing the "speak with a stutter" instruction with the free TTS options is
likely a minor effect, since the disfluency markers (e.g. `"I-I-I want to
sss-see you"`) are already textually present before TTS — the instruction
in the paper's design mainly nudges acoustic delivery on top of that, per
the paper's own description of that stage.

## 6. Next steps (pending advisor input)

1. Decide: continue with paid OpenAI stack once billing is resolved, or
   switch development to the free stack (Ollama/Groq + edge-tts/Piper) and
   reserve paid APIs for a final, larger-scale data generation run.
2. Once unblocked, run `--dry-run-text` on the sample sentences to
   qualitatively check disfluency quality before generating any audio.
3. Generate a small pilot batch (~20–50 sentences) end-to-end (text + audio)
   for a first listen/quality check.
4. Scale to the full source corpus (mirroring the paper's use of Mozilla
   Common Voice — need to decide on an analogous English source corpus).
5. Feed the manifest into a Whisper fine-tuning script, following the
   paper's Section 3.2–4.1 setup.

## 7. Repository structure

```
stutter_pipeline/
├── config.py                  # models, disfluency taxonomy, sampling weights, TTS settings
├── sampler.py                 # per-sentence disfluency label sampling
├── disfluency_generator.py    # batched LLM call for text disfluency injection
├── tts_generator.py           # TTS synthesis (voice/speed sampling, stutter instruction)
├── pipeline.py                 # end-to-end orchestration + manifest writer
├── requirements.txt
├── data/
│   └── sentences.txt          # example fluent English sentences
└── outputs/                   # generated at runtime
    ├── text/disfluent_dataset.json
    ├── audio_clean/
    ├── audio_stutter/
    └── manifest.csv
```