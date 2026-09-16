"""HW#2 - Generate Restaurant Sentences with Trigram Model (슬라이드 p.13).

바이그램 생성(p.9)을 트라이그램으로 확장한다.
    P(w_i+2 | w_i, w_i+1) = C(w_i, w_i+1, w_i+2) / C(w_i, w_i+1)

핵심 차이
    - 상태(context)가 단어 1개가 아니라 (앞앞 단어, 앞 단어) 2개다.
    - 첫 단어는 트라이그램 context가 아직 없으므로 (<s>, w) 바이그램에서 뽑는다.
    - 해당 context의 트라이그램이 하나도 없으면 바이그램으로 backoff 한다.

실행:
    python hw02_trigram.py
"""

import random
from collections import Counter

from ngram import BOS, EOS, NgramModel, build_corpus

# -------------------------------------------------------------------------
# 1) 직접 구현 버전 (제출용으로 이 부분을 읽고 이해할 것)
# -------------------------------------------------------------------------
corpus = build_corpus()

bigram_counts = Counter()
trigram_counts = Counter()
for sentence in corpus:
    for i in range(len(sentence) - 1):
        bigram_counts[(sentence[i], sentence[i + 1])] += 1
    for i in range(len(sentence) - 2):
        w1, w2, w3 = sentence[i:i + 3]
        trigram_counts[(w1, w2, w3)] += 1


def sample_next_word_bigram(previous_word):
    """바이그램 카운트를 가중치로 다음 단어 샘플링."""
    candidates = []
    weights = []
    for (w1, w2), count in bigram_counts.items():
        if w1 == previous_word:
            candidates.append(w2)
            weights.append(count)
    if not candidates:
        return EOS
    return random.choices(candidates, weights=weights, k=1)[0]


def sample_next_word_trigram(w1, w2):
    """(w1, w2) context에서 다음 단어 샘플링. context가 없으면 바이그램 backoff."""
    candidates = []
    weights = []
    for (a, b, c), count in trigram_counts.items():
        if a == w1 and b == w2:
            candidates.append(c)
            weights.append(count)
    if not candidates:
        return sample_next_word_bigram(w2)
    return random.choices(candidates, weights=weights, k=1)[0]


def generate_sentence_trigram(max_length=15):
    """<s>에서 시작해 트라이그램으로 문장을 생성한다."""
    first = sample_next_word_bigram(BOS)   # 첫 단어는 바이그램으로
    if first == EOS:
        return ""

    prev1, prev2 = BOS, first
    output = [first]
    for _ in range(max_length - 1):
        next_word = sample_next_word_trigram(prev1, prev2)
        if next_word == EOS:
            break
        output.append(next_word)
        prev1, prev2 = prev2, next_word
    return " ".join(output)


# -------------------------------------------------------------------------
# 2) 비교: 바이그램 생성 vs 트라이그램 생성
# -------------------------------------------------------------------------
def main():
    random.seed(42)  # 재현 가능하게. 제출 전에 여러 seed로 돌려볼 것.

    model = NgramModel(corpus)

    print("=== Bigram 생성 (p.9) ===")
    for _ in range(10):
        print(" ", model.generate_sentence())

    print("\n=== Trigram 생성 (HW#2, 직접 구현) ===")
    for _ in range(10):
        print(" ", generate_sentence_trigram())

    print("\n=== 생성 문장의 perplexity 비교 ===")
    for _ in range(5):
        s = generate_sentence_trigram()
        print(f"  {s!r:45s} PP={model.perplexity(s, mode='backoff'):.3f}")

    print("\n=== 관찰 포인트 ===")
    print("  - 트라이그램은 학습 문장을 거의 그대로 재현한다(데이터가 15문장뿐이라).")
    print("  - context가 길수록 카운트가 희박해져 zero-probability가 심해진다.")
    print("  - 그래서 backoff / interpolation 같은 smoothing이 필요하다.")


if __name__ == "__main__":
    main()
