# faster-whisper — Phrase Bias 포크 문서 (dev-docs)

이 저장소는 [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)의 **포크**(branch `feature/phrase-bias`)이며,
STT 도메인 용어 **signed phrase bias**를 위한 커스텀 레이어를 추가한다.

> 상세 설계 문서는 `dev-docs/`에 둔다. README는 이 포크의 사용자 진입점에 필요한 phrase bias 사용법만 갱신하고, upstream PR용 브랜치에서는 필요 시 분리한다.

---

## 이 포크가 하는 일 (한 줄)

사용자 친화적 **phrase bias config(JSON)** 를 받아, Whisper 토크나이저로 토큰화하여
**CTranslate2 Whisper phrase bias DTO**(`ctranslate2.models.PhraseBias`)로 컴파일하고
`WhisperModel` 생성 시 주입한다. 실제 logit bias 적용은 **CT2 디코더**가 한다.

```
phrase_bias config(JSON)  ──compile──▶  CT2 PhraseBias DTO  ──inject──▶  CT2 Whisper 디코더
   (이 레포: phrase_bias.py)              (token ids + signed step_bias)  (reverse-trie, logit += bias)
```

이 레포는 **컴파일러 + 통합**만 담당한다. trie 구축·logit 가산·clamp 는 **CT2 쪽**(`feature/whisper-phrase-bias`).

---

## 문서 색인

| 문서 | 내용 |
|---|---|
| [architecture.md](architecture.md) | 컴파일러 내부(2-variant 토큰화·schedule 수학)·config 스키마·CT2 핸드오프·통합 지점 |
| [p5-negative-bias-impl-plan.md](p5-negative-bias-impl-plan.md) | signed negative bias 구현 범위·테스트·문서 갱신 계획 |
| [production-readiness.md](production-readiness.md) | 프로덕션 준비도 감사(이 레포 측 findings와 해소 상태) |

## 코드 맵

| 파일 | 역할 |
|---|---|
| `faster_whisper/phrase_bias.py` | **컴파일러**. config 로드·검증, 2-variant 토큰화, uniform/ramp schedule, CT2 DTO 변환 |
| `faster_whisper/transcribe.py` | **통합**. `WhisperModel.__init__`에서 컴파일·주입(`710-733`), hf_tokenizer 선로드(`715`) |
| `examples/phrase_bias_config.json` | config 예시 |
| `benchmark/phrase_bias_ab.py` | A/B 벤치 |
| `tests/test_phrase_bias.py` | 단위/통합 테스트 |

## 상위(cross-repo) 설계 = CT2 SSOT

phrase bias **전체 설계**(trie 의미론, prefix 중첩 합산, clamp, 레이어 계약, 토큰화 결정)는
CT2 포크의 단일 진실 공급원에 통합돼 있다:

- `CTranslate2/dev-docs/SSOT.md` — 통합 설계 문서
- `CTranslate2/dev-docs/production-readiness-review.md` — 두 레포 종합 프로덕션 감사

이 레포 문서는 **faster-whisper 측 레이어**(컴파일러·통합)에 한정하고, 교차 설계는 위를 참조한다.

## 현재 상태 (2026-06-04)

- 컴파일러·통합·테스트 구현 완료. CT2 `feature/whisper-phrase-bias` 와 짝으로 동작.
- 평가 하니스: 별도 레포 `phrase-bias-eval/` (eval·self-evolve 튜닝).
- 프로덕션 감사 결과: **수학 핵심은 정확**, 운영 관찰성·입력 검증은 hardening에서 보강됨 → [production-readiness.md](production-readiness.md).
