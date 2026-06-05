# Phase 5 Implementation Plan — Signed Negative Bias

## Goal

`phrase_bias_config`의 같은 `terms[].bias` 필드에서 음수를 허용한다.
양수는 도메인 용어 boost, 음수는 자주 생기는 오인식 후보 soft suppress로 사용한다.

## Scope

- positive/negative를 별도 config로 나누지 않는다.
- negative도 leading-space/non-leading-space 두 variant를 모두 컴파일한다.
- 1-token phrase는 계속 skip한다. start token negative bias는 별도 과제로 둔다.
- hard block/suppress는 범위 밖이다.
- 모델 로딩 테스트는 이번 단계에서 제외한다.

## Tasks

1. Parser/schema
   - `default_total_bias`, `min_total_bias`, `max_total_bias`, term `bias`를 finite signed float로 읽는다.
   - 기본값은 `default_total_bias=5.0`, `min_total_bias=-5.0`, `max_total_bias=5.0`, `max_step_bias=2.0`.
2. Scheduler
   - uniform은 `[-max_step_bias,+max_step_bias]`로 step clamp한다.
   - ramp는 negative cumulative delta와 increment를 보존한다.
3. Tests
   - negative uniform: 두 surface path가 생성되고 `step_bias=-2.0`으로 clamp되는지 확인.
   - negative ramp: 음수 increment가 유지되는지 확인.
4. Docs/examples
   - README, `dev-docs/architecture.md`, example JSON을 signed semantics로 갱신한다.

## Verification

- `tests/test_phrase_bias.py` 비모델 테스트
- `ruff check faster_whisper/phrase_bias.py faster_whisper/transcribe.py tests/test_phrase_bias.py`
- `py_compile` for touched Python files
