"""
LLM-based text disfluency injection for English transcripts.

This is a direct English adaptation of the batch GPT-4o-mini prompt
described in the paper: it takes a batch of {original_text, disfluency_type}
pairs and returns {original_text, disfluent_text, disfluency_type} triples,
where only words consistent with the assigned label(s) are perturbed.
"""

import json
import time
from typing import List, Dict, Any

from openai import OpenAI

from config import TEXT_MODEL, TEXT_BATCH_SIZE, TEXT_MAX_RETRIES, TEXT_TEMPERATURE

SYSTEM_PROMPT = """You are a model that converts fluent English text into stuttered text (speech disfluency).

Task
Receive input as a batch of fluent English sentences (original_text), each paired with a pre-determined
disfluency_type list.
Transform each original_text into disfluent text according to the given disfluency_type(s).
Do not change the disfluency_type(s) that were given.
If a disfluency type such as "repetition", "prolongation", or "interjection" is present in disfluency_type,
make sure at least one word in the sentence exhibits that type of disfluency.

Disfluency types you may be asked to apply
1. Repetition: repetition of a syllable, word, or short phrase
   (e.g., "I" becomes "I-I-I", "want" becomes "wa-wa-want").
2. Interjection: insertion of filler words such as "um", "uh", "er", "like", "you know" between words.
3. Prolongation: lengthening of the initial letter(s) of a word, roughly three to five repeated letters
   (e.g., "something" becomes "sss-something", "want" becomes "wwww-want").

Rules
- Preserve the meaning of the sentence and the original word order. Do not add new information.
- Only lengthen the BEGINNING of a word for prolongation, e.g. "sss-something", never "somethingg".
- Choose only a few words per sentence to make disfluent -- never the entire sentence.
- Do not place an interjection as the very last token of the sentence.
- If disfluency_type includes a category, the output MUST contain at least one instance of that category.
- Keep "original_text" and "disfluency_type" in the output IDENTICAL to what was given in the input --
  do not paraphrase, correct, or otherwise modify them.
- Output ONLY a JSON array, no prose, no markdown code fences.

Input format
[
  {"original_text": "I want to eat chicken", "disfluency_type": ["repetition"]},
  {"original_text": "they want to discuss the new project plan", "disfluency_type": ["repetition", "interjection"]},
  {"original_text": "I received a long message from an old friend", "disfluency_type": ["repetition", "prolongation"]}
]

Output format
[
  {"original_text": "I want to eat chicken", "disfluent_text": "I-I-I want to eat ch-ch-chicken", "disfluency_type": ["repetition"]},
  {"original_text": "they want to discuss the new project plan", "disfluent_text": "th-th-they um want to discuss the new pr-pr-project plan", "disfluency_type": ["repetition", "interjection"]},
  {"original_text": "I received a long message from an old friend", "disfluent_text": "I-I-I rrr-received a mmm-message from an old friend", "disfluency_type": ["repetition", "prolongation"]}
]

Reminder: "original_text" and "disfluency_type" in every output object must exactly match the corresponding
input object. Only "disfluent_text" is new."""


class DisfluencyGenerator:
    def __init__(self, api_key: str = None, model: str = TEXT_MODEL):
        self.client = OpenAI(api_key=api_key)  # falls back to OPENAI_API_KEY env var
        self.model = model

    def _call_llm(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        user_payload = json.dumps(
            [{"original_text": s, "disfluency_type": t} for s, t in batch],
            ensure_ascii=False,
        )

        last_err = None
        for attempt in range(1, TEXT_MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=TEXT_TEMPERATURE,
                    response_format={"type": "json_object"} if "gpt-4o" in self.model else None,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
                raw = _strip_code_fences(raw)
                parsed = json.loads(raw)

                # response_format=json_object requires a top-level JSON object,
                # so the model may wrap the array; handle both shapes.
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            parsed = v
                            break

                if not isinstance(parsed, list):
                    raise ValueError(f"Expected a JSON list, got {type(parsed)}")

                self._validate(batch, parsed)
                return parsed

            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = 2 ** attempt
                print(f"[disfluency_generator] attempt {attempt} failed: {e}. retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"LLM batch failed after {TEXT_MAX_RETRIES} attempts: {last_err}")

    @staticmethod
    def _validate(batch, parsed):
        if len(parsed) != len(batch):
            raise ValueError(f"Expected {len(batch)} items back, got {len(parsed)}")
        for (orig, types), item in zip(batch, parsed):
            for key in ("original_text", "disfluent_text", "disfluency_type"):
                if key not in item:
                    raise ValueError(f"Missing '{key}' in LLM output item: {item}")
            if item["original_text"].strip() != orig.strip():
                raise ValueError(
                    f"original_text drifted: expected {orig!r}, got {item['original_text']!r}"
                )

    def generate(self, labeled_sentences: List[tuple]) -> List[Dict[str, Any]]:
        """
        labeled_sentences: list of (sentence:str, disfluency_types:List[str])
        Returns: list of dicts with original_text, disfluent_text, disfluency_type
        """
        results = []
        for i in range(0, len(labeled_sentences), TEXT_BATCH_SIZE):
            batch = labeled_sentences[i : i + TEXT_BATCH_SIZE]
            print(f"[disfluency_generator] processing batch {i // TEXT_BATCH_SIZE + 1} "
                  f"({len(batch)} sentences)")
            results.extend(self._call_llm(batch))
        return results


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()
