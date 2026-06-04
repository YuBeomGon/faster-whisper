"""
faster_whisper/phrase_bias.py
Compile user-facing phrase bias config into CTranslate2 Whisper phrase bias DTOs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import ctranslate2


@dataclass(frozen=True)
class CompiledPhraseBiasPath:
    ids: List[int]
    step_bias: float
    min_prefix_len: int = 1


@dataclass(frozen=True)
class CompiledPhraseBias:
    surface: str
    token_paths: List[CompiledPhraseBiasPath]


ConfigInput = Optional[Union[str, Mapping[str, Any]]]


_TOP_LEVEL_KEYS = {
    "version",
    "enabled",
    "default_total_bias",
    "bias_schedule",
    "min_total_bias",
    "max_total_bias",
    "max_step_bias",
    "min_prefix_len",
    "terms",
}
_TERM_KEYS = {"text", "bias", "aliases", "schedule", "min_prefix_len"}


def load_phrase_bias_config(config: ConfigInput) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as file:
            loaded = json.load(file)
    elif isinstance(config, Mapping):
        loaded = dict(config)
    else:
        raise TypeError("phrase_bias_config must be a path, a mapping, or None")

    if not isinstance(loaded, dict):
        raise ValueError("phrase_bias_config must be a JSON object")
    unknown = set(loaded) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError("Unknown phrase_bias_config keys: %s" % sorted(unknown))
    if loaded.get("version") != 1:
        raise ValueError("phrase_bias_config.version must be 1")
    if "terms" not in loaded or not isinstance(loaded["terms"], list):
        raise ValueError("phrase_bias_config.terms must be a list")
    return loaded


def compile_phrase_bias_config(
    config: ConfigInput,
    tokenizer,
) -> List[CompiledPhraseBias]:
    loaded = load_phrase_bias_config(config)
    if not loaded or loaded.get("enabled", True) is False:
        return []

    min_total_bias = float(loaded.get("min_total_bias", 0.1))
    max_total_bias = float(loaded.get("max_total_bias", 1.5))
    max_step_bias = float(loaded.get("max_step_bias", 0.5))
    default_bias = float(loaded.get("default_total_bias", 0.5))
    default_schedule = loaded.get("bias_schedule", "uniform")
    default_min_prefix_len = int(loaded.get("min_prefix_len", 1))

    compiled: List[CompiledPhraseBias] = []
    seen_surfaces = set()
    for term in loaded["terms"]:
        if not isinstance(term, dict):
            raise ValueError("Each phrase bias term must be an object")
        unknown = set(term) - _TERM_KEYS
        if unknown:
            raise ValueError("Unknown phrase bias term keys: %s" % sorted(unknown))

        text = str(term.get("text", "")).strip()
        if not text:
            raise ValueError("Phrase bias term text must be non-empty")

        surfaces = [text]
        for alias in term.get("aliases", []) or []:
            alias_text = str(alias).strip()
            if alias_text:
                surfaces.append(alias_text)

        total_bias = _clamp(
            float(term.get("bias", default_bias)),
            min_total_bias,
            max_total_bias,
        )
        schedule = term.get("schedule", default_schedule)
        min_prefix_len = int(term.get("min_prefix_len", default_min_prefix_len))

        for surface in surfaces:
            if surface in seen_surfaces:
                continue
            seen_surfaces.add(surface)
            token_paths = _compile_surface(
                surface,
                tokenizer,
                total_bias,
                schedule,
                max_step_bias,
                min_prefix_len,
            )
            if token_paths:
                compiled.append(CompiledPhraseBias(surface=surface, token_paths=token_paths))
    return compiled


def _compile_surface(
    surface: str,
    tokenizer,
    total_bias: float,
    schedule: str,
    max_step_bias: float,
    min_prefix_len: int,
) -> List[CompiledPhraseBiasPath]:
    paths: List[CompiledPhraseBiasPath] = []
    seen_paths = set()
    for variant in (" " + surface, surface):
        ids = _encode_clean_ids(tokenizer, variant)
        if len(ids) < 2:
            continue
        if tokenizer.decode(ids) != variant:
            continue
        key = tuple(ids)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        paths.extend(_expand_schedule(ids, total_bias, schedule, max_step_bias, min_prefix_len))
    return paths


def _encode_clean_ids(tokenizer, surface: str) -> List[int]:
    ids = list(tokenizer.encode(surface, add_special_tokens=False).ids)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    return [token_id for token_id in ids if token_id < eot_id]


def _expand_schedule(
    ids: Sequence[int],
    total_bias: float,
    schedule: str,
    max_step_bias: float,
    min_prefix_len: int,
) -> List[CompiledPhraseBiasPath]:
    continuation_count = len(ids) - 1
    if continuation_count <= 0:
        return []

    if schedule == "uniform":
        step_bias = min(total_bias / continuation_count, max_step_bias)
        return [
            CompiledPhraseBiasPath(
                ids=list(ids),
                step_bias=step_bias,
                min_prefix_len=min_prefix_len,
            )
        ]

    if schedule == "ramp":
        weight_sum = continuation_count * (continuation_count + 1) / 2
        deltas = [
            min(total_bias * (i + 1) / weight_sum, max_step_bias)
            for i in range(continuation_count)
        ]
        paths = []
        previous = 0.0
        for index, delta in enumerate(deltas, start=1):
            increment = delta - previous
            previous = delta
            if increment <= 0:
                continue
            paths.append(
                CompiledPhraseBiasPath(
                    ids=list(ids),
                    step_bias=increment,
                    min_prefix_len=max(min_prefix_len, index),
                )
            )
        return paths

    raise ValueError("Unsupported phrase bias schedule: %s" % schedule)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def to_ctranslate2_phrase_biases(
    compiled: Iterable[CompiledPhraseBias],
) -> List[ctranslate2.models.PhraseBias]:
    phrase_biases = []
    for bias in compiled:
        token_paths = [
            ctranslate2.models.PhraseBiasPath(
                ids=path.ids,
                step_bias=path.step_bias,
                min_prefix_len=path.min_prefix_len,
            )
            for path in bias.token_paths
        ]
        if token_paths:
            phrase_biases.append(ctranslate2.models.PhraseBias(token_paths=token_paths))
    return phrase_biases
