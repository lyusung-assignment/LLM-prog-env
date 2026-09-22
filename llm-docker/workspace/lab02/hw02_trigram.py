"""HW#2 - Generate restaurant sentences with a trigram model (slide 13).

Extends the bigram generator from slide 9 to trigrams:
    P(w_i+2 | w_i, w_i+1) = C(w_i, w_i+1, w_i+2) / C(w_i, w_i+1)

What changes
    - The context is two words (w_i, w_i+1), not one.
    - The first word has no trigram context yet, so it is drawn from the
      (<s>, w) bigrams.
    - When a context has no trigram at all, back off to the bigram.

Run:
    python hw02_trigram.py
"""

import random
from collections import Counter

from ngram import BOS, EOS, NgramModel, build_corpus

# -------------------------------------------------------------------------
# 1) Trigram counts and sampling
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
    """Sample the next word, weighting candidates by their bigram counts."""
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
    """Sample on the context (w1, w2), backing off to the bigram when unseen."""
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
    """Start at <s> and generate a sentence with the trigram model."""
    first = sample_next_word_bigram(BOS)   # no trigram context yet
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
# 2) Compare bigram generation against trigram generation
# -------------------------------------------------------------------------
def main():
    random.seed(42)  # reproducible; try other seeds before submitting

    model = NgramModel(corpus)

    print("=== Bigram generation (slide 9) ===")
    for _ in range(10):
        print(" ", model.generate_sentence())

    print("\n=== Trigram generation (HW#2, hand-written) ===")
    for _ in range(10):
        print(" ", generate_sentence_trigram())

    print("\n=== Perplexity of the generated sentences ===")
    for _ in range(5):
        s = generate_sentence_trigram()
        print(f"  {s!r:45s} PP={model.perplexity(s, mode='backoff'):.3f}")

    print("\n=== What to notice ===")
    print("  - The trigram model mostly replays the training sentences,")
    print("    because there are only 15 of them.")
    print("  - A longer context means sparser counts and more zero")
    print("    probabilities.")
    print("  - That is exactly why smoothing (backoff / interpolation)")
    print("    is needed.")


if __name__ == "__main__":
    main()
