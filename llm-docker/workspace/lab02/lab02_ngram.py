"""Lab 2 - N-gram models, following the slides in order.

Work through this file top to bottom during class. Each step maps one-to-one
onto a page of Lab2_Ngram.pdf.

Run:
    python lab02_ngram.py
"""

import math
import random
from collections import Counter

# =========================================================================
# Slide 3  A small Berkeley-style corpus
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
# Slide 4  Add sentence boundaries
#          Generation also has to learn how sentences begin (<s>) and end (</s>).
# =========================================================================
def tokenize(sentence):
    return ["<s>"] + sentence.lower().split() + ["</s>"]


corpus = [tokenize(s) for s in sentences]


def step_tokenize():
    print("\n[slide 4] Tokenization")
    for sentence in corpus[:3]:
        print(sentence)


# =========================================================================
# Slide 5  Unigram counts        P(w) = C(w) / N
# =========================================================================
unigram_counts = Counter()
for sentence in corpus:
    unigram_counts.update(sentence)

total_words = sum(unigram_counts.values())


def unigram_prob(word):
    return unigram_counts[word] / total_words


def step_unigram():
    print("\n[slide 5] Unigram counts")
    print(unigram_counts.most_common(10))
    print("total_words =", total_words)
    print("P(want) =", unigram_prob("want"))
    print("P(food) =", unigram_prob("food"))


# =========================================================================
# Slide 6  Bigram counts         (a Counter keyed by tuples)
# =========================================================================
bigram_counts = Counter()
for sentence in corpus:
    for i in range(len(sentence) - 1):
        bigram = (sentence[i], sentence[i + 1])
        bigram_counts[bigram] += 1


def step_bigram_counts():
    print("\n[slide 6] Bigram counts (top 5)")
    for bigram, count in bigram_counts.most_common(5):
        print(bigram, count)


# =========================================================================
# Slide 7  Bigram probability    MLE: P(w_i+1 | w_i) = C(w_i, w_i+1) / C(w_i)
# =========================================================================
def bigram_prob(previous_word, word):
    numerator = bigram_counts[(previous_word, word)]
    denominator = unigram_counts[previous_word]
    if denominator == 0:
        return 0
    return numerator / denominator


def step_bigram_prob():
    print("\n[slide 7] Bigram probability")
    print("P(want | i) =", bigram_prob("i", "want"))
    print("P(food | chinese) =", bigram_prob("chinese", "food"))
    print("P(restaurant | a) =", bigram_prob("a", "restaurant"))


# =========================================================================
# Slide 8  Next-word prediction
# =========================================================================
def predict_bigram(previous_word, top_k=3):
    candidates = []
    for (w1, w2), count in bigram_counts.items():
        if w1 == previous_word:
            probability = bigram_prob(w1, w2)
            candidates.append((w2, probability))
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]


def step_predict():
    print("\n[slide 8] Next word prediction")
    print("want ->", predict_bigram("want"))
    print("i    ->", predict_bigram("i"))
    print("<s>  ->", predict_bigram("<s>"))


# =========================================================================
# Extra) Zero-probability problem / backoff / interpolation
#        A bigram missing from the training corpus has probability 0, which
#        drags the probability of the whole sentence down to 0.
# =========================================================================
def backoff_prob(previous_word, word, alpha=0.4):
    """Stupid backoff: fall back to the unigram, discounted by alpha."""
    p = bigram_prob(previous_word, word)
    if p > 0:
        return p
    return alpha * unigram_prob(word)


def interpolation_prob(previous_word, word, lambda1=0.7, lambda2=0.3):
    """Linear interpolation: mix bigram and unigram (the lambdas sum to 1)."""
    return lambda1 * bigram_prob(previous_word, word) + lambda2 * unigram_prob(word)


def step_smoothing():
    print("\n[extra] Zero probability / backoff / interpolation")
    print("MLE           P(pizza | want) =", bigram_prob("want", "pizza"))
    print("Backoff       P(food  | want) =", backoff_prob("want", "food"))
    print("Backoff       P(chinese | italian) =", backoff_prob("italian", "chinese"))
    print("Interpolation P(food  | want) =", interpolation_prob("want", "food"))


# =========================================================================
# Extra) Sentence probability   P(W) = prod P(w_i | w_i-1)
# =========================================================================
def sentence_prob(sentence, prob=bigram_prob):
    tokens = tokenize(sentence)
    p = 1.0
    for i in range(len(tokens) - 1):
        p *= prob(tokens[i], tokens[i + 1])
    return p


def step_sentence_prob():
    print("\n[extra] Sentence probability")
    for s in ["i want chinese food", "i want korean food"]:
        print(s)
        print("   MLE     =", sentence_prob(s))
        print("   backoff =", sentence_prob(s, prob=backoff_prob))


# =========================================================================
# Slide 11  Perplexity   PP(W) = exp( -1/N * sum log P(w_i | context) )
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
    print("\n[slide 11] Perplexity")
    test_sentences = [
        "i want chinese food",
        "i want thai food",
        "restaurant chinese want i",
    ]
    for s in test_sentences:
        print(s, perplexity(s))


# =========================================================================
# Slide 9  Generate restaurant sentences (bigram sampling)
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
    print("\n[slide 9] Generate restaurant sentences (bigram)")
    for _ in range(10):
        print(generate_sentence())


# =========================================================================
# Slide 12  Trigram probability
#           MLE: P(w_i+2 | w_i, w_i+1) = C(w1, w2, w3) / C(w1, w2)
#           Note: the slide's bigram__counts is a typo (one underscore).
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
    print("\n[slide 12] Trigram probability")
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
    print("\n[slide 13] Trigram generation is HW#2 -> see hw02_trigram.py")


if __name__ == "__main__":
    main()
