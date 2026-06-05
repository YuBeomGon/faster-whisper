# Architecture — faster-whisper Phrase Bias 컴파일러

이 레포 측 레이어(컴파일러 + 통합)의 동작. 교차 설계(trie·clamp·logit 가산)는 `CTranslate2/dev-docs/SSOT.md`.

---

## 1. 데이터 흐름

```
config(JSON/dict)
  └─ load_phrase_bias_config()      스키마 검증(키 화이트리스트, version==1, terms is list)
       └─ compile_phrase_bias_config(config, tokenizer)
            ├─ term별: surfaces = [text] + aliases, signed bias = clamp(bias, min_total, max_total)
            └─ surface별(전역 dedup):
                 └─ _compile_surface()
                      ├─ 2-variant: (" "+surface, surface)
                      ├─ _encode_clean_ids(): encode → eot 이상 id strip
                      ├─ 필터: len(ids) >= 2,  decode(ids) == variant (roundtrip)
                      └─ _expand_schedule(): uniform | ramp → CompiledPhraseBiasPath[]
       └─ to_ctranslate2_phrase_biases()   → ctranslate2.models.PhraseBias[]
            └─ WhisperModel.__init__ → model_kwargs["phrase_biases"]  (CT2로 주입)
```

DTO 계층:
- `CompiledPhraseBiasPath(ids: List[int], step_bias: float, min_prefix_len: int=1)` — 이 레포 내부 표현
- `ctranslate2.models.PhraseBiasPath(ids, step_bias, min_prefix_len)` — CT2 바인딩(핸드오프 형식)
- `ctranslate2.models.PhraseBias(token_paths=[...])` — CT2가 받는 최종 DTO

---

## 2. Config 스키마

상위 키 (`_TOP_LEVEL_KEYS`):

| 키 | 기본 | 의미 |
|---|---|---|
| `version` | **필수 = 1** | 스키마 버전(다르면 ValueError) |
| `enabled` | `true` | `false`면 컴파일 결과 `[]`(no-op) |
| `default_total_bias` | `5.0` | term `bias` 미지정 시 |
| `bias_schedule` | `"uniform"` | term `schedule` 미지정 시 (`uniform`\|`ramp`) |
| `min_total_bias` | `-5.0` | term bias clamp 하한 |
| `max_total_bias` | `5.0` | term bias clamp 상한 |
| `max_step_bias` | `2.0` | step당 bias 절댓값 상한(분배 후 per-step clamp는 -2.0~+2.0) |
| `min_prefix_len` | `1` | bias 발화에 필요한 최소 prefix 길이 |
| `terms` | **필수, list** | 용어 배열 |

term 키 (`_TERM_KEYS`):

| 키 | 의미 |
|---|---|
| `text` | **필수, non-empty**. 도메인 용어 표면형 |
| `bias` | signed total bias(미지정 시 `default_total_bias`), `[min_total,max_total]`로 clamp. 양수는 boost, 음수는 soft suppress |
| `aliases` | 추가 표면형 리스트(예 "Comfy UI" ← "ComfyUI") |
| `schedule` | per-term schedule override |
| `min_prefix_len` | per-term override |

> 알 수 없는 키는 ValueError(상위·term 양쪽 화이트리스트). 예시는 `examples/phrase_bias_config.json`.

---

## 3. 2-variant 토큰화 (왜 2개인가)

BPE 토크나이저는 **선행 공백**을 토큰에 인코딩한다("문장 중간 등장" vs "문두 등장"이 다른 토큰열).
도메인 phrase는 문장 아무 위치에서나 나올 수 있으므로 두 표면형을 모두 컴파일한다:

- `" " + surface` — 문장 중간(선행 공백 포함) 등장
- `surface` — 문두/공백 없는 등장

각 variant에 대해 (`_compile_surface`, `_encode_clean_ids`):
1. `tokenizer.encode(variant, add_special_tokens=False).ids`
2. `<|endoftext|>` **이상 id 제거**(특수/타임스탬프/언어 태그 배제) — `token_id < eot_id`
3. `len(ids) >= 2` 요구 (1-토큰 용어는 continuation이 없어 skip)
4. **roundtrip 체크** `tokenizer.decode(ids) == variant` (불일치 variant 드롭)
5. `ids` 튜플로 path dedup, surface는 전역 `seen_surfaces`로 dedup

> 3·4단계 드롭은 `logger.warning`으로 이유를 남긴다(too-few-tokens, roundtrip mismatch, duplicate token path, min_prefix_len 초과).

---

## 4. Schedule 수학

`continuation_count = len(ids) - 1` (첫 토큰엔 bias 미적용, continuation step에만).

### uniform
모든 continuation step에 동일 bias:
```
step_bias = clamp(total_bias / continuation_count, -max_step_bias, +max_step_bias)
```
단일 path. CT2 trie가 prefix 진행마다 `step_bias`를 가산.

### ramp (뒤로 갈수록 강하게)
가중치 `weight_sum = n(n+1)/2`, 위치 i의 누적 목표 `delta[i] = clamp(total*(i+1)/weight_sum, -max_step_bias, +max_step_bias)`.
**누적 증분(increment)** 으로 인코딩 → CT2 trie의 가산 누적(`out[token] += step`)과 결합되어
위치 j의 실효 signed bias가 `total * j / weight_sum`로 정확히 복원, 총합 = `total_bias`.
각 increment는 `min_prefix_len = max(min_prefix_len, index)`로 발화 위치를 제한.

> 감사 결과 ramp/uniform 수학은 **정확**함이 CT2 trie와 교차검증됨(production-readiness.md 참고 항목).

---

## 5. CT2 핸드오프 & 통합 (transcribe.py)

`WhisperModel.__init__(..., phrase_bias_config=None)`:

- **충돌 가드**(`710-711`): `phrase_bias_config` 와 raw `phrase_biases` kwarg 동시 사용 금지.
- **토크나이저 선로드**(`715`, `_load_hf_tokenizer`): 컴파일에 토크나이저가 필요해 **모델 생성 전** 로드.
  - `_fallback_tokenizer_name`(`624`)은 이름의 `.en`/`-en` 접미사로 다국어 여부 추정(폴백 한정, production-readiness.md M5).
- **컴파일·주입**(`721-733`): `compile_phrase_bias_config` → `to_ctranslate2_phrase_biases` → `model_kwargs["phrase_biases"]`.
  - 빈 결과면 kwarg 미설정. `self.phrase_bias_config`(원본)·`self.compiled_phrase_biases` 보관.

CT2 측이 받은 뒤(`CTranslate2/dev-docs/SSOT.md`):
- 모델 로드 시 1회 **reverse-prefix trie** 빌드(immutable, `shared_ptr<const>`로 공유).
- 디코드 매 step `PhraseBiasProcessor`가 현재 시퀀스 끝에서 역방향 lookup → 매칭 path의 `step_bias` **합산** → `clamp(sum, -max_token_delta, +max_token_delta)`(기본 ±2.0) → 해당 token logit에 가산.
- prefix 중첩 시 bias 합산은 CT2에서 일어난다(이 레포는 path만 제공).

---

## 6. no-op 의미론

| 입력 | 동작 |
|---|---|
| `phrase_bias_config=None` | 컴파일 안 함. phrase bias 미적용 |
| `enabled: false` | `compile_*` 가 `[]` 반환 → kwarg 미설정 |
| 모든 term 드롭(1-토큰/roundtrip) | `[]` → kwarg 미설정 (⚠️ "config 없음"과 구분 불가, production-readiness.md H4) |
| 정상 term 존재 | `phrase_biases=[...]` 주입 |
