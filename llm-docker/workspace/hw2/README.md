# HW#2 - N-gram Language Models

`Lab2_Ngram.pdf` 과제 (LLM Programming, 상명대학교 컴퓨터과학과 강상욱 교수).
제출기한 9월 22일 23:59, 제출물은 **PDF 한 개**.

| 문제 | 내용 | 코드 |
| --- | --- | --- |
| 1 | Trigram LM 구현 후 문장 5개 생성 | `problem1()` |
| 2 | Unigram/Bigram/Trigram perplexity 비교 (테스트 문장 5개 이상, 훈련 문장 제외) | `problem2()` |
| - | 스무딩 상수 k에 따른 변화 (추가 확인) | `smoothing_sweep()` |

## 실행

```bash
python hw2_ngram.py
```

표준 라이브러리(`math`, `random`, `collections`)만 사용한다. `lab02/` 를 import
하지 않는 단독 파일이라 이 파일 하나만 제출하면 된다.

```bash
docker compose exec llm python /workspace/hw2/hw2_ngram.py
```

## 설계에서 신경 쓴 부분

1. **스무딩.** MLE만 쓰면 학습에 없는 n-gram 하나 때문에 문장 확률이 0,
   perplexity가 `inf` 가 되어 세 모델을 비교할 수 없다. 그래서 add-k(Laplace)를
   적용했다.
   `P(w | ctx) = (C(ctx, w) + k) / (C(ctx) + k * V)`
   테스트 문장에만 나오는 단어는 `<unk>` 로 매핑하고 V에 한 칸을 준다.
2. **비교의 공정성.** n차 모델은 `<s>` 를 n-1개 붙여서 패딩한다. 그래야 세 모델이
   테스트 문장의 **같은 토큰**(w_1 ... w_m, `</s>`)을 예측하게 되고, 분모 N이
   같아져 perplexity를 그대로 비교할 수 있다.
3. **생성은 MLE 분포로.** 스무딩된 분포로 샘플링하면 어휘 전체에 확률이 조금씩
   퍼져 엉뚱한 단어가 섞인다. 생성은 원래 카운트로 하고, 못 본 2-word 컨텍스트는
   bigram → unigram 순으로 backoff 한다.

## 결과 (k=1.0, seed=42)

생성 문장 5개 — 5개 전부 훈련 문장과 동일하다. 훈련 문장이 15개뿐이라 두 단어
컨텍스트의 다음 단어가 거의 유일하게 결정되기 때문이다.

테스트 문장 8개에 대한 corpus perplexity:

| | unigram | bigram | trigram |
| --- | --- | --- | --- |
| 테스트 문장 | 20.89 | **10.13** | 12.65 |
| 훈련 문장 | 18.31 | 7.73 | 8.74 |

- unigram이 가장 나쁘다. 컨텍스트를 무시하므로 단어마다 거의 `log V` 만큼의
  비용을 낸다.
- **bigram이 가장 좋다.** 한 단어 컨텍스트만으로도 후보가 크게 줄고, 필요한
  bigram(`want to`, `italian food` 등)은 훈련에서 이미 봤다.
- trigram은 훈련 문장에서는 bigram보다 낫지만 테스트 문장에서는 더 나쁘다.
  못 본 2-word 컨텍스트가 add-k의 균등분포 `1/V` 로 떨어지면서 비용이 커진다.
  컨텍스트를 늘리는 것은 데이터가 충분할 때만 이득이라는 sparsity/overfitting
  trade-off 를 그대로 보여준다.
- k를 0.001 ~ 1.0 으로 바꿔봐도 bigram < trigram 순서는 유지된다. 즉 이 결과는
  스무딩 상수가 아니라 코퍼스 크기에서 오는 성질이다.

## 테스트 문장

훈련 문장 15개와 겹치지 않는 8개를 썼고, 코드에서 `assert` 로 겹치지 않음을
검증한다. 마지막 문장에는 미등록 단어(`korean`)를 일부러 넣어 OOV 처리를
확인한다.
