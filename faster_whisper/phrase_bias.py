"""
faster_whisper/phrase_bias.py
Compile user-facing phrase bias config into CTranslate2 Whisper phrase bias DTOs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import ctranslate2

logger = logging.getLogger(__name__)


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

    min_total_bias = _read_finite_float(loaded.get("min_total_bias", 0.0), "min_total_bias")
    max_total_bias = _read_finite_float(loaded.get("max_total_bias", 5.0), "max_total_bias")
    if min_total_bias > max_total_bias:
        raise ValueError("min_total_bias must be <= max_total_bias")
    max_step_bias = _read_nonnegative_float(loaded.get("max_step_bias", 2.0), "max_step_bias")
    if max_step_bias <= 0:
        raise ValueError("max_step_bias must be > 0")
    default_bias = _read_finite_float(loaded.get("default_total_bias", 5.0), "default_total_bias")
    default_schedule = loaded.get("bias_schedule", "uniform")
    default_min_prefix_len = _read_min_prefix_len(loaded.get("min_prefix_len", 1), "min_prefix_len")

    compiled: List[CompiledPhraseBias] = []
    seen_surfaces = set()
    seen_token_paths = set()
    for term in loaded["terms"]:
        if not isinstance(term, dict):
            raise ValueError("Each phrase bias term must be an object")
        unknown = set(term) - _TERM_KEYS
        if unknown:
            raise ValueError("Unknown phrase bias term keys: %s" % sorted(unknown))

        text_value = term.get("text", "")
        if not isinstance(text_value, str):
            raise ValueError("Phrase bias term text must be a string")
        text = text_value.strip()
        if not text:
            raise ValueError("Phrase bias term text must be non-empty")

        surfaces = [text]
        aliases = term.get("aliases", [])
        if aliases is None:
            aliases = []
        if not isinstance(aliases, list):
            raise ValueError("Phrase bias term aliases must be a list")
        for alias in aliases:
            if not isinstance(alias, str):
                raise ValueError("Phrase bias term aliases must contain strings")
            alias_text = alias.strip()
            if alias_text:
                surfaces.append(alias_text)

        total_bias = _clamp(
            _read_finite_float(term.get("bias", default_bias), "bias"),
            min_total_bias,
            max_total_bias,
        )
        schedule = term.get("schedule", default_schedule)
        if not isinstance(schedule, str):
            raise ValueError("Phrase bias schedule must be a string")
        min_prefix_len = _read_min_prefix_len(
            term.get("min_prefix_len", default_min_prefix_len),
            "min_prefix_len",
        )

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
                seen_token_paths,
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
    seen_token_paths: Optional[set] = None,
) -> List[CompiledPhraseBiasPath]:
    paths: List[CompiledPhraseBiasPath] = []
    seen_paths = set()
    for variant in (" " + surface, surface):
        ids = _encode_clean_ids(tokenizer, variant)
        if len(ids) < 2:
            logger.warning(
                "Dropped phrase bias variant %r for surface %r: too few tokens",
                variant,
                surface,
            )
            continue
        decoded = tokenizer.decode(ids)
        if decoded != variant:
            logger.warning(
                "Dropped phrase bias variant %r for surface %r: roundtrip mismatch decoded %r",
                variant,
                surface,
                decoded,
            )
            continue
        key = tuple(ids)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        if seen_token_paths is not None and key in seen_token_paths:
            logger.warning(
                "Dropped phrase bias variant %r for surface %r: duplicate token path",
                variant,
                surface,
            )
            continue
        expanded = _expand_schedule(ids, total_bias, schedule, max_step_bias, min_prefix_len)
        if not expanded:
            logger.warning(
                "Dropped phrase bias variant %r for surface %r: min_prefix_len exceeds continuation count",
                variant,
                surface,
            )
            continue
        if seen_token_paths is not None:
            seen_token_paths.add(key)
        paths.extend(expanded)
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
    if min_prefix_len > continuation_count:
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


def _read_nonnegative_float(value: Any, name: str) -> float:
    number = _read_finite_float(value, name)
    if number < 0:
        raise ValueError("%s must be >= 0" % name)
    return number


def _read_finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number" % name) from None
    if not math.isfinite(number):
        raise ValueError("%s must be a finite number" % name)
    return number


def _read_min_prefix_len(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer" % name) from None
    if number < 1:
        raise ValueError("%s must be >= 1" % name)
    return number


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
