# Lab 2 - N-gram Models

`Lab2_Ngram.pdf` (LLM Programming, 상명대학교 컴퓨터과학과 강상욱 교수) 실습 코드.

## 파일

| 파일 | 내용 |
| --- | --- |
| `lab02_ngram.py` | 슬라이드 p.3 ~ p.12를 순서대로 따라가는 실습 스크립트 |
| `ngram.py` | 같은 내용을 `NgramModel` 클래스로 정리한 재사용 모듈 |
| `hw02_trigram.py` | HW#2 - 트라이그램 모델로 문장 생성 (슬라이드 p.13) |

## 실행

```bash
python lab02_ngram.py     # 실습 전체 흐름
python hw02_trigram.py    # 과제: 트라이그램 생성
```

표준 라이브러리(`collections`, `math`, `random`)만 쓰므로 lab01처럼 별도
conda 환경(`environment.yml`)을 만들 필요가 없다. 컨테이너의 공통 `llm` 환경에서
바로 돌아간다.

```bash
docker compose exec llm python /workspace/lab02/lab02_ngram.py
docker compose exec llm python /workspace/lab02/hw02_trigram.py
```

## 슬라이드 ↔ 코드 대응

| 슬라이드 | 내용 | 코드 |
| --- | --- | --- |
| p.3 | Restaurant corpus (15문장) | `SENTENCES` |
| p.4 | 문장 경계 `<s>`, `</s>` 추가 | `tokenize()` |
| p.5 | Unigram counts, `P(w) = C(w)/N` | `unigram_prob()` |
| p.6 | Bigram counts (tuple key) | `bigram_counts` |
| p.7 | Bigram MLE `C(w1,w2)/C(w1)` | `bigram_prob()` |
| p.8 | Next word prediction | `predict_bigram()` |
| p.9 | 문장 생성 (분포에서 샘플링) | `sample_next_word()`, `generate_sentence()` |
| p.11 | Perplexity (로그 확률 사용) | `perplexity()` |
| p.12 | Trigram MLE | `trigram_prob()` |
| p.13 | Trigram 문장 생성 | **HW#2** → `hw02_trigram.py` |

슬라이드 흐름에는 있지만 코드가 없는 항목(zero-probability, backoff,
interpolation, sentence probability)은 `step_smoothing()`,
`step_sentence_prob()` 및 `NgramModel.backoff_prob()` /
`interpolation_prob()` 로 추가해 두었다.

## 메모

- 슬라이드 p.12의 `bigram__counts` 는 언더바 2개 오타다. 코드에서는 `bigram_counts` 로 수정했다.
- MLE만 쓰면 학습에 없는 bigram 하나 때문에 문장 확률이 0, perplexity가 `inf` 가 된다.
  (`"restaurant chinese want i"` 로 확인 가능)
- 코퍼스가 15문장뿐이라 트라이그램은 학습 문장을 거의 그대로 외운다.
  n이 커질수록 sparsity가 심해진다는 점을 직접 확인하는 것이 이 실습의 목적.
