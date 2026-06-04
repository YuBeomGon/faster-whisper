"""
tests/test_phrase_bias.py
Tests for faster-whisper phrase bias config compilation.
"""

import json

import pytest

from faster_whisper.phrase_bias import (
    CompiledPhraseBiasPath,
    compile_phrase_bias_config,
    load_phrase_bias_config,
)


class FakeWhisperTokenizer:
    def __init__(self):
        self._eot = 50000
        self._encoded = {
            " 트랜스포머": [10, 11, 12],
            "트랜스포머": [20, 11, 12],
            " Transformer": [30, 31, 32, 33],
            "Transformer": [40, 31, 32, 33],
            " 짧음": [50],
            "짧음": [51],
            " special": [60, 62, 50001],
            "special": [61, 62, 50001],
        }
        self._decoded = {
            (10, 11, 12): " 트랜스포머",
            (20, 11, 12): "트랜스포머",
            (30, 31, 32, 33): " Transformer",
            (40, 31, 32, 33): "Transformer",
            (50,): " 짧음",
            (51,): "짧음",
            (60, 62): " special",
            (61, 62): "special",
        }

    def token_to_id(self, token):
        if token == "<|endoftext|>":
            return self._eot
        raise KeyError(token)

    def encode(self, text, add_special_tokens=False):
        class Encoded:
            def __init__(self, ids):
                self.ids = ids

        return Encoded(list(self._encoded[text]))

    def decode(self, ids):
        return self._decoded[tuple(ids)]


def test_load_phrase_bias_config_from_path(tmp_path):
    path = tmp_path / "phrase_bias.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "default_total_bias": 0.5,
                "terms": [{"text": "트랜스포머"}],
            }
        ),
        encoding="utf-8",
    )

    config = load_phrase_bias_config(str(path))

    assert config["version"] == 1
    assert config["terms"] == [{"text": "트랜스포머"}]


def test_compile_uniform_two_surface_paths():
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "default_total_bias": 0.6,
            "bias_schedule": "uniform",
            "terms": [{"text": "트랜스포머"}],
        },
        FakeWhisperTokenizer(),
    )

    assert len(compiled) == 1
    assert compiled[0].surface == "트랜스포머"
    assert compiled[0].token_paths == [
        CompiledPhraseBiasPath(ids=[10, 11, 12], step_bias=0.3, min_prefix_len=1),
        CompiledPhraseBiasPath(ids=[20, 11, 12], step_bias=0.3, min_prefix_len=1),
    ]
