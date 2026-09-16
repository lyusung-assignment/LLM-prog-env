"""Lab 2 - Ngram Models 실습 스크립트 (슬라이드 순서 그대로).

수업 중에는 이 파일을 위에서부터 한 단계씩 따라가면 된다.
각 단계는 슬라이드 페이지 번호와 1:1로 대응한다.

실행:
    python lab02_ngram.py
"""

import math
import random
from collections import Counter

# =========================================================================
# p.3  A Small Berkeley-style Corpus
# =========================================================================
sentences = [
    "i want chinese food",
    "i want thai food",
    "i want italian food",
    "i want a restaurant",
    "i want to eat",
    "i want to eat chinese food",
    "i want to eat thai food",
    "i am looking for chinese food",
    "i am looking for a restaurant",
    "can you find a chinese restaurant",
    "can you find a thai restaurant",
    "can you find an italian restaurant",
    "tell me about chinese restaurants",
    "tell me about thai restaurants",
    "where can i eat chinese food",
]


# =========================================================================
# p.4  Add Sentence Boundaries
#      문장이 어떻게 시작(<s>)하고 끝나는지(</s>)도 학습해야 생성이 가능하다.
# =========================================================================
def tokenize(sentence):
    return ["<s>"] + sentence.lower().split() + ["</s>"]


corpus = [tokenize(s) for s in sentences]


def step_tokenize():
    print("\n[p.4] Tokenization")
    for sentence in corpus[:3]:
        print(sentence)


# =========================================================================
# p.5  Unigram Counts        P(w) = C(w) / N
# =========================================================================
unigram_counts = Counter()
for sentence in corpus:
    unigram_counts.update(sentence)

total_words = sum(unigram_counts.values())


def unigram_prob(word):
    return unigram_counts[word] / total_words


def step_unigram():
    print("\n[p.5] Unigram counts")
    print(unigram_counts.most_common(10))
    print("total_words =", total_words)
    print("P(want) =", unigram_prob("want"))
    print("P(food) =", unigram_prob("food"))


# =========================================================================
# p.6  Bigram Counts         (tuple을 key로 쓰는 Counter)
# =========================================================================
bigram_counts = Counter()
for sentence in corpus:
    for i in range(len(sentence) - 1):
        bigram = (sentence[i], sentence[i + 1])
        bigram_counts[bigram] += 1


def step_bigram_counts():
    print("\n[p.6] Bigram counts (top 5)")
    for bigram, count in bigram_counts.most_common(5):
        print(bigram, count)


# =========================================================================
# p.7  Bigram Probability    MLE: P(w_i+1 | w_i) = C(w_i, w_i+1) / C(w_i)
# =========================================================================
def bigram_prob(previous_word, word):
    numerator = bigram_counts[(previous_word, word)]
    denominator = unigram_counts[previous_word]
    if denominator == 0:
        return 0
    return numerator / denominator


def step_bigram_prob():
    print("\n[p.7] Bigram probability")
    print("P(want | i) =", bigram_prob("i", "want"))
    print("P(food | chinese) =", bigram_prob("chinese", "food"))
    print("P(restaurant | a) =", bigram_prob("a", "restaurant"))


# =========================================================================
# p.8  Next Word Prediction
# =========================================================================
def predict_bigram(previous_word, top_k=3):
    candidates = []
    for (w1, w2), count in bigram_counts.items():
        if w1 == previous_word:
            probability = bigram_prob(w1, w2)
            candidates.append((w2, probability))
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]


def step_predict():
    print("\n[p.8] Next word prediction")
    print("want ->", predict_bigram("want"))
    print("i    ->", predict_bigram("i"))
    print("<s>  ->", predict_bigram("<s>"))


# =========================================================================
# 보충) Zero-probability problem / Backoff / Interpolation
#      학습 코퍼스에 없는 bigram은 확률이 0 -> 문장 확률 전체가 0이 된다.
# =========================================================================
def backoff_prob(previous_word, word, alpha=0.4):
    """Stupid backoff: bigram이 0이면 unigram으로 내려가되 alpha로 할인한다."""
    p = bigram_prob(previous_word, word)
    if p > 0:
        return p
    return alpha * unigram_prob(word)


def interpolation_prob(previous_word, word, lambda1=0.7, lambda2=0.3):
    """Linear interpolation: bigram과 unigram을 섞는다. (lambda 합 = 1)"""
    return lambda1 * bigram_prob(previous_word, word) + lambda2 * unigram_prob(word)


def step_smoothing():
    print("\n[보충] Zero probability / Backoff / Interpolation")
    print("MLE           P(pizza | want) =", bigram_prob("want", "pizza"))
    print("Backoff       P(food  | want) =", backoff_prob("want", "food"))
    print("Backoff       P(chinese | italian) =", backoff_prob("italian", "chinese"))
    print("Interpolation P(food  | want) =", interpolation_prob("want", "food"))


# =========================================================================
# 보충) Sentence Probability   P(W) = prod P(w_i | w_i-1)
# =========================================================================
def sentence_prob(sentence, prob=bigram_prob):
    tokens = tokenize(sentence)
    p = 1.0
    for i in range(len(tokens) - 1):
        p *= prob(tokens[i], tokens[i + 1])
    return p


def step_sentence_prob():
    print("\n[보충] Sentence probability")
    for s in ["i want chinese food", "i want korean food"]:
        print(s)
        print("   MLE     =", sentence_prob(s))
        print("   backoff =", sentence_prob(s, prob=backoff_prob))


# =========================================================================
# p.11  Perplexity   PP(W) = exp( -1/N * sum log P(w_i | context) )
# =========================================================================
def perplexity(sentence):
    tokens = tokenize(sentence)
    log_prob = 0
    n = len(tokens) - 1
    for i in range(n):
        p = bigram_prob(tokens[i], tokens[i + 1])
        if p == 0:
            return float("inf")
        log_prob += math.log(p)
    return math.exp(-log_prob / n)


def step_perplexity():
    print("\n[p.11] Perplexity")
    test_sentences = [
        "i want chinese food",
        "i want thai food",
        "restaurant chinese want i",
    ]
    for s in test_sentences:
        print(s, perplexity(s))


# =========================================================================
# p.9  Generate Restaurant Sentences (bigram sampling)
# =========================================================================
def sample_next_word(previous_word):
    candidates = []
    weights = []
    for (w1, w2), count in bigram_counts.items():
        if w1 == previous_word:
            candidates.append(w2)
            weights.append(count)
    if not candidates:
        return "</s>"
    return random.choices(candidates, weights=weights, k=1)[0]


def generate_sentence(max_length=15):
    current = "<s>"
    output = []
    for _ in range(max_length):
        next_word = sample_next_word(current)
        if next_word == "</s>":
            break
        output.append(next_word)
        current = next_word
    return " ".join(output)


def step_generate():
    print("\n[p.9] Generate restaurant sentences (bigram)")
    for _ in range(10):
        print(generate_sentence())


# =========================================================================
# p.12  Trigram Probability
#       MLE: P(w_i+2 | w_i, w_i+1) = C(w1, w2, w3) / C(w1, w2)
#       ※ 슬라이드의 bigram__counts 는 오타 (언더바 1개가 맞음)
# =========================================================================
trigram_counts = Counter()
for sentence in corpus:
    for i in range(len(sentence) - 2):
        w1, w2, w3 = sentence[i:i + 3]
        trigram_counts[(w1, w2, w3)] += 1


def trigram_prob(w1, w2, w3):
    numerator = trigram_counts[(w1, w2, w3)]
    denominator = bigram_counts[(w1, w2)]
    if denominator == 0:
        return 0
    return numerator / denominator


def step_trigram():
    print("\n[p.12] Trigram probability")
    print("P(chinese | i, want) =", trigram_prob("i", "want", "chinese"))
    print("P(to      | i, want) =", trigram_prob("i", "want", "to"))
    print("top-5 trigrams:", trigram_counts.most_common(5))


def main():
    step_tokenize()
    step_unigram()
    step_bigram_counts()
    step_bigram_prob()
    step_predict()
    step_smoothing()
    step_sentence_prob()
    step_perplexity()
    step_generate()
    step_trigram()
    print("\n[p.13] Trigram 문장 생성은 HW#2 -> hw02_trigram.py 참고")


if __name__ == "__main__":
    main()
