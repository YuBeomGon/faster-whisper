"""
tests/test_phrase_bias.py
Tests for faster-whisper phrase bias config compilation.
"""

import json
import types

import pytest

from faster_whisper.phrase_bias import (
    CompiledPhraseBiasPath,
    compile_phrase_bias_config,
    load_phrase_bias_config,
    to_ctranslate2_phrase_biases,
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


class CollidingWhisperTokenizer(FakeWhisperTokenizer):
    def encode(self, text, add_special_tokens=False):
        self._last_text = text

        class Encoded:
            def __init__(self, ids):
                self.ids = ids

        return Encoded([70, 71])

    def decode(self, ids):
        return self._last_text


def test_load_phrase_bias_config_from_path(tmp_path):
    path = tmp_path / "phrase_bias.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "default_total_bias": 5.0,
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


def test_compile_default_biases_use_stronger_limits():
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "terms": [{"text": "트랜스포머"}],
        },
        FakeWhisperTokenizer(),
    )

    assert compiled[0].token_paths == [
        CompiledPhraseBiasPath(ids=[10, 11, 12], step_bias=2.0, min_prefix_len=1),
        CompiledPhraseBiasPath(ids=[20, 11, 12], step_bias=2.0, min_prefix_len=1),
    ]


def test_compile_default_max_total_bias_clamps_at_five():
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "max_step_bias": 10.0,
            "terms": [{"text": "트랜스포머", "bias": 7.0}],
        },
        FakeWhisperTokenizer(),
    )

    assert compiled[0].token_paths == [
        CompiledPhraseBiasPath(ids=[10, 11, 12], step_bias=2.5, min_prefix_len=1),
        CompiledPhraseBiasPath(ids=[20, 11, 12], step_bias=2.5, min_prefix_len=1),
    ]


def test_compile_negative_bias_uses_two_surface_paths():
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "terms": [{"text": "트랜스포머", "bias": -5.0}],
        },
        FakeWhisperTokenizer(),
    )

    assert len(compiled) == 1
    assert compiled[0].surface == "트랜스포머"
    assert compiled[0].token_paths == [
        CompiledPhraseBiasPath(ids=[10, 11, 12], step_bias=-2.0, min_prefix_len=1),
        CompiledPhraseBiasPath(ids=[20, 11, 12], step_bias=-2.0, min_prefix_len=1),
    ]


def test_compile_ramp_schedule_as_cumulative_min_prefix_paths():
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "bias_schedule": "ramp",
            "terms": [{"text": "Transformer", "bias": 0.6}],
        },
        FakeWhisperTokenizer(),
    )

    # step_bias is a computed float (cumulative ramp increment), so compare it
    # with approx; ids/min_prefix_len are exact. (Plan used exact dataclass
    # equality which is fragile for float arithmetic, e.g. 0.2 - 0.1.)
    paths = compiled[0].token_paths
    assert len(paths) >= 3
    for index, path in enumerate(paths[:3], start=1):
        assert path.ids == [30, 31, 32, 33]
        assert path.min_prefix_len == index
        assert path.step_bias == pytest.approx(0.1, abs=1e-9)


def test_compile_negative_ramp_schedule_keeps_negative_increments():
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "bias_schedule": "ramp",
            "terms": [{"text": "Transformer", "bias": -0.6}],
        },
        FakeWhisperTokenizer(),
    )

    paths = compiled[0].token_paths
    assert len(paths) >= 3
    for index, path in enumerate(paths[:3], start=1):
        assert path.ids == [30, 31, 32, 33]
        assert path.min_prefix_len == index
        assert path.step_bias == pytest.approx(-0.1, abs=1e-9)


def test_compile_skips_one_token_surfaces():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": True, "terms": [{"text": "짧음", "bias": 0.5}]},
        FakeWhisperTokenizer(),
    )

    assert compiled == []


def test_compile_strips_special_tokens_and_keeps_roundtrip():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": True, "terms": [{"text": "special", "bias": 0.5}]},
        FakeWhisperTokenizer(),
    )

    assert compiled[0].token_paths == [
        CompiledPhraseBiasPath(ids=[60, 62], step_bias=0.5, min_prefix_len=1),
        CompiledPhraseBiasPath(ids=[61, 62], step_bias=0.5, min_prefix_len=1),
    ]


def test_enabled_false_compiles_empty():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": False, "terms": [{"text": "트랜스포머"}]},
        FakeWhisperTokenizer(),
    )

    assert compiled == []


def test_unknown_config_key_raises():
    with pytest.raises(ValueError, match="Unknown phrase_bias_config keys"):
        compile_phrase_bias_config(
            {"version": 1, "terms": [], "typo": True},
            FakeWhisperTokenizer(),
        )


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {"version": 1, "min_total_bias": 1.0, "max_total_bias": 0.5, "terms": []},
            "min_total_bias must be <= max_total_bias",
        ),
        (
            {"version": 1, "max_step_bias": 0.0, "terms": []},
            "max_step_bias must be > 0",
        ),
        (
            {"version": 1, "min_prefix_len": 0, "terms": []},
            "min_prefix_len must be >= 1",
        ),
        (
            {"version": 1, "terms": [{"text": "트랜스포머", "aliases": "별칭"}]},
            "aliases must be a list",
        ),
    ],
)
def test_invalid_phrase_bias_config_values_raise(config, message):
    with pytest.raises(ValueError, match=message):
        compile_phrase_bias_config(config, FakeWhisperTokenizer())


def test_compile_warns_when_surface_is_dropped(caplog):
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": True, "terms": [{"text": "짧음", "bias": 0.5}]},
        FakeWhisperTokenizer(),
    )

    assert compiled == []
    assert "Dropped phrase bias variant" in caplog.text
    assert "too few tokens" in caplog.text


def test_compile_deduplicates_token_paths_globally(caplog):
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "terms": [{"text": "Alpha"}, {"text": "Beta"}],
        },
        CollidingWhisperTokenizer(),
    )

    assert len(compiled) == 1
    assert compiled[0].surface == "Alpha"
    assert compiled[0].token_paths == [
        CompiledPhraseBiasPath(ids=[70, 71], step_bias=2.0, min_prefix_len=1)
    ]
    assert "duplicate token path" in caplog.text


def test_to_ctranslate2_phrase_biases_roundtrip():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": True, "terms": [{"text": "트랜스포머", "bias": 0.6}]},
        FakeWhisperTokenizer(),
    )

    ct2_biases = to_ctranslate2_phrase_biases(compiled)

    assert len(ct2_biases) == 1
    assert [list(path.ids) for path in ct2_biases[0].token_paths] == [
        [10, 11, 12],
        [20, 11, 12],
    ]


def test_to_ctranslate2_phrase_biases_preserves_negative_step_bias():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": True, "terms": [{"text": "트랜스포머", "bias": -0.6}]},
        FakeWhisperTokenizer(),
    )

    ct2_biases = to_ctranslate2_phrase_biases(compiled)

    assert len(ct2_biases) == 1
    assert [path.step_bias for path in ct2_biases[0].token_paths] == pytest.approx(
        [-0.3, -0.3],
        abs=1e-6,
    )


def test_whisper_model_init_passes_compiled_phrase_biases(monkeypatch, tmp_path):
    from faster_whisper import transcribe
    from faster_whisper.transcribe import WhisperModel

    captured = {}

    class FakeWhisper:
        is_multilingual = False

        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["kwargs"] = kwargs

    monkeypatch.setattr(transcribe.ctranslate2.models, "Whisper", FakeWhisper)
    monkeypatch.setattr(
        transcribe, "_load_hf_tokenizer", lambda *args, **kwargs: FakeWhisperTokenizer()
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    WhisperModel(
        str(model_dir),
        device="cpu",
        phrase_bias_config={
            "version": 1,
            "enabled": True,
            "terms": [{"text": "트랜스포머", "bias": 0.6}],
        },
    )

    assert "phrase_biases" in captured["kwargs"]
    assert len(captured["kwargs"]["phrase_biases"]) == 1
    assert [list(path.ids) for path in captured["kwargs"]["phrase_biases"][0].token_paths] == [
        [10, 11, 12],
        [20, 11, 12],
    ]


def test_phrase_bias_config_conflicts_with_raw_ct2_phrase_biases(tmp_path):
    from faster_whisper.transcribe import WhisperModel

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="phrase_bias_config cannot be used with phrase_biases"):
        WhisperModel(
            str(model_dir),
            phrase_bias_config={"version": 1, "enabled": False, "terms": []},
            phrase_biases=[],
        )


def test_phrase_bias_config_requires_ct2_phrase_bias_support(monkeypatch, tmp_path):
    from faster_whisper import transcribe
    from faster_whisper.transcribe import WhisperModel

    models = types.SimpleNamespace(Whisper=object)
    monkeypatch.setattr(transcribe.ctranslate2, "models", models)
    monkeypatch.setattr(
        transcribe, "_load_hf_tokenizer", lambda *args, **kwargs: FakeWhisperTokenizer()
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(RuntimeError, match="does not support phrase_bias_config"):
        WhisperModel(
            str(model_dir),
            phrase_bias_config={"version": 1, "terms": [{"text": "트랜스포머"}]},
        )


def test_fallback_tokenizer_prefers_ct2_vocabulary_over_model_name(tmp_path):
    from faster_whisper.transcribe import _fallback_tokenizer_name

    model_dir = tmp_path / "custom-english-model"
    model_dir.mkdir()
    (model_dir / "vocabulary.json").write_text(
        json.dumps(["hello", "<|endoftext|>"]),
        encoding="utf-8",
    )

    assert (
        _fallback_tokenizer_name("not-ending-with-en", str(model_dir))
        == "openai/whisper-tiny.en"
    )

    (model_dir / "vocabulary.json").write_text(
        json.dumps(["hello", "<|en|>", "<|ko|>", "<|endoftext|>"]),
        encoding="utf-8",
    )

    assert (
        _fallback_tokenizer_name("custom.en", str(model_dir))
        == "openai/whisper-tiny"
    )


def test_actual_whisper_tokenizer_compiles_leading_and_non_leading_paths():
    from faster_whisper import WhisperModel

    model = WhisperModel("tiny")
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "terms": [{"text": "transformer", "bias": 0.5}],
        },
        model.hf_tokenizer,
    )

    assert len(compiled) == 1
    decoded_paths = [model.hf_tokenizer.decode(path.ids) for path in compiled[0].token_paths]
    assert " transformer" in decoded_paths
    assert "transformer" in decoded_paths
    assert all(len(path.ids) >= 2 for path in compiled[0].token_paths)
