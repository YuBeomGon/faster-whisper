"""
benchmark/phrase_bias_ab.py
Run phrase-bias A/B evaluation over an audio manifest.
"""

from __future__ import annotations

import argparse
import json
import time

from faster_whisper import WhisperModel


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def contains_any(text: str, terms):
    normalized = normalize(text)
    return any(normalize(term) in normalized for term in terms)


def transcribe_one(model, audio_path):
    start = time.perf_counter()
    segments, _ = model.transcribe(audio_path, beam_size=5, temperature=0.0)
    text = "".join(segment.text for segment in segments)
    latency = time.perf_counter() - start
    return text, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phrase-bias-config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="default")
    args = parser.parse_args()

    baseline = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    biased = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        phrase_bias_config=args.phrase_bias_config,
    )

    rows = []
    with open(args.manifest, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    total = 0
    recall_base = 0
    recall_bias = 0
    insertion_base = 0
    insertion_bias = 0
    latency_base = 0.0
    latency_bias = 0.0

    for row in rows:
        total += 1
        terms = row["terms"]
        expected_present = bool(row.get("expected_present", True))
        base_text, base_latency = transcribe_one(baseline, row["audio"])
        bias_text, bias_latency = transcribe_one(biased, row["audio"])
        base_has = contains_any(base_text, terms)
        bias_has = contains_any(bias_text, terms)

        if expected_present:
            recall_base += int(base_has)
            recall_bias += int(bias_has)
        else:
            insertion_base += int(base_has)
            insertion_bias += int(bias_has)

        latency_base += base_latency
        latency_bias += bias_latency
        print(
            json.dumps(
                {
                    "audio": row["audio"],
                    "terms": terms,
                    "baseline_has_term": base_has,
                    "biased_has_term": bias_has,
                    "baseline_text": base_text,
                    "biased_text": bias_text,
                    "baseline_latency": base_latency,
                    "biased_latency": bias_latency,
                },
                ensure_ascii=False,
            )
        )

    print(
        json.dumps(
            {
                "total": total,
                "recall_base": recall_base,
                "recall_bias": recall_bias,
                "insertion_base": insertion_base,
                "insertion_bias": insertion_bias,
                "avg_latency_base": latency_base / max(total, 1),
                "avg_latency_bias": latency_bias / max(total, 1),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
