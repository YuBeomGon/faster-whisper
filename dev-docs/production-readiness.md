# 프로덕션 준비도 — faster-whisper Phrase Bias 측

> 감사일 2026-06-04 · 대상 `feature/phrase-bias` · 정적 감사(코드 정독, file:line 근거)
> 업데이트 2026-06-05: B1/H1/H4 hardening과 signed negative bias가 반영됐다. 이 문서는 감사 기준과 해소 상태를 함께 남긴다.
> 두 레포 종합본은 `CTranslate2/dev-docs/production-readiness-review.md`. 이 문서는 **faster-whisper 측 발췌**.

---

## TL;DR

**컴파일러의 수학 핵심(ramp/uniform, trie 인코딩)은 정확하고 CT2 바인딩 스키마와 형식이 일치한다.**
감사 당시 운영 리스크는 코어가 아니라 **무음 실패(observability)** 와 **입력 검증 부재(robustness)** 였다.
2026-06-05 기준 드롭 경고, 수치 검증, 구버전 CT2 capability check는 구현됐다.

| 영역 | 평가 |
|---|---|
| schedule/토큰화 수학 | 🟢 정확(CT2 trie와 교차검증) |
| 운영 관찰성 | 🟢 드롭 warning 추가 |
| 입력 검증 | 🟢 signed bias 범위·min_prefix_len·min/max 역전 검증 |
| 통합/버전 호환 | 🟢 구버전 CT2 capability 선검증 |
| 테스트(실토크나이저) | 🟠 1케이스만 |
| upstream PR 준비도 | 🟠 중(미흡) |

---

## Findings

### 🟢 B1. 무음 실패 — phrase가 경고 없이 드롭됨 (해소됨)
`phrase_bias.py:122,137-154` · 파일 전체 로깅 0건(확인).
3중 드롭 경로가 전부 무음:
1. 1-토큰 용어(`len(ids)<2`) → 두 variant 스킵 → term 자체 누락
2. roundtrip 불일치(`decode(ids)!=variant`) → variant 드롭
3. 과도한 `min_prefix_len` → 전 위치 제외 → bias 통째 무력화

**영향**: 용어 50개 등록 시 일부가 아무 경고 없이 미적용. 컴파일 결과 surface 목록 INFO 로그조차 없어 진단 불가.
**상태**: 드롭 surface/variant를 `logger.warning`으로 기록한다(too-few-tokens / roundtrip-mismatch / duplicate token path / min_prefix_len 초과).

### 🟢 H1. 입력 수치 검증 부재 (해소됨)
`phrase_bias.py:76-79,102-108,195`
- `min_prefix_len` 하한/상한 없음 → `0`(의미 불명), 음수(난해한 pybind 에러, `needs-runtime-check`), 거대값(silent no-op).
- `min_total_bias > max_total_bias`(오타) → `_clamp`가 모든 bias를 한 값으로 붕괴.
- 음수 bias는 signed soft suppress 용도로 지원한다(`min_total_bias=-5.0`, `max_total_bias=5.0`, `max_step_bias=2.0` 기본).

**상태**: `min_prefix_len >= 1`, `min_total_bias <= max_total_bias`, `max_step_bias > 0`, finite number 검증이 들어갔다.

### 🟢 H4. 구버전 CT2 capability 선검증 부재 (해소됨)
`transcribe.py:710-733`
구버전 CT2(=`ctranslate2.models.PhraseBias` 미존재)에 `phrase_biases` kwarg를 던지면
"unexpected keyword argument"류 난해한 에러. README의 "PhraseBias 지원 빌드 필요"를 코드가 선검사 안 함.
**상태**: `hasattr(ctranslate2.models, "PhraseBias")` 선검사 후 친절한 에러를 낸다.

### 🟠 M4. 실토크나이저 roundtrip 회귀 커버리지 1케이스 (medium)
`phrase_bias.py:151-154`, test `:228`(actual whisper tokenizer = "transformer" 1건)
- 위험: 선행 공백 BPE 규칙상 비띄어쓰기 variant가 `decode` 복원 실패하는 토크나이저/언어(한국어 등)에서 조용히 드롭(→B1).
- 숫자/하이픈/혼합("GPT-4","v3.5")·유니코드 roundtrip 미검증.
**수정**: 실제 Whisper 토크나이저로 한국어·숫자·하이픈·이모지·혼합 phrase roundtrip 회귀 테스트 추가.

### 🟠 M5. 다국어 판정이 모델속성→문자열 추정으로 후퇴 (medium)
`transcribe.py:624-628,715-719`
기존엔 `self.model.is_multilingual`(권위 속성)로 `.en` 토크나이저 선택. 이제 컴파일을 모델 생성 *전*에 해야 해서
`_fallback_tokenizer_name`이 디렉터리명 `.en`/`-en` 접미사로 추정. `.en`로 안 끝나는 영어전용 로컬모델은
다국어 tiny 토크나이저로 폴백 → 잘못된 token id로 컴파일. **단 `tokenizer.json` 부재 시에만 작동**(실모델 대부분 무관).
**수정**: 컴파일을 모델 생성 *후*로 옮겨 `is_multilingual` 사용, 또는 폴백 시 경고.

### 🟠 M6. 통합 테스트가 기계적 flip만 (medium)
`test_phrase_bias.py:87-113` — `_select_flippable_target`가 런타임 탐색 후 못 찾으면 skip, `step_bias=50.0`로 강제 flip.
"충분히 큰 bias면 토큰 바뀜"만 확인 — **실제 도메인 recall 향상(이 포크 목적)을 검증하지 않음**. 모델/transformers 버전 변하면 skip으로 무력화.
**수정**: no-op(None/`[]`) 동등성 검증에 집중(이 부분은 양호), recall은 별도 eval 하니스(`phrase-bias-eval/`)로.

### 🟢 Low / 정보
- **L1** 동시성 낮은 위험: `phrase_biases`는 모델 생성 시 1회 컴파일→CT2 immutable trie로 공유. `BatchedInferencePipeline`도 자체 가변 상태 없음. `self.phrase_bias_config`에 원본 dict 보관(read-only). 방어적으로 `deepcopy` 무방.
- **L2** 타입힌트: `compile_phrase_bias_config(config, tokenizer)`의 `tokenizer` 힌트 없음(duck-typed). Protocol 힌트 권장.

---

## 긍정 (확정)

- **ramp/uniform 수학 정확**: 누적 증분 인코딩이 CT2 trie 가산 누적과 결합해 위치별 실효 boost `total*j/weight_sum`, 총합 `total_bias` 정확 복원(`CTranslate2/src/decoding_utils.cc:71-98,122-160` 교차검증).
- **바인딩 스키마 일치**: `PhraseBiasPath(ids, step_bias=, min_prefix_len=)`, kw_only 인자 형식 정확.
- **dedup 건전**: 2-variant(`seen_paths`)·surface(`seen_surfaces`) 중복 제거 논리적.
- **스키마 검증 강함**: 상위·term 키 화이트리스트, version·terms 타입 체크.

---

## Upstream PR 준비도: 중(미흡)

SYSTRAN 후방호환·범용성 관점:
1. [done] B1 무음 드롭 로깅
2. [done] H1 수치 불변식 검증
3. [done] H4 구버전 CT2 capability 선검증 + 친절한 에러
4. [med] M4 실토크나이저 다언어/숫자/혼합 roundtrip 회귀 테스트

위 해소 시 PR 제출 가능 수준. 현재 테스트는 fake 토크나이저 위주라 실토크나이저 커버리지가 1케이스뿐.
