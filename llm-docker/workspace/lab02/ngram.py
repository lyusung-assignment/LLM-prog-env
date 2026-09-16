"""Lab 2 - N-gram models, packaged as a reusable module.

The same code as the slides (Lab2_Ngram.pdf), reorganized so the lab script
(lab02_ngram.py) and the homework (hw02_trigram.py) can import it.

Conceptual flow
    Restaurant sentences -> Tokenization -> Unigram/Bigram/Trigram counts
    -> MLE probabilities -> Next-word prediction -> Zero-probability problem
    -> Backoff -> Interpolation -> Sentence probability -> Perplexity
    -> Sentence generation
"""

from __future__ import annotations

import math
import random
from collections import Counter

BOS = "<s>"   # beginning of sentence
EOS = "</s>"  # end of sentence

# --- A small Berkeley-style corpus (slide 3) ------------------------------
SENTENCES = [
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


# --- Tokenization (slide 4) -----------------------------------------------
def tokenize(sentence):
    """Lowercase, split on whitespace, and wrap in sentence-boundary tokens."""
    return [BOS] + sentence.lower().split() + [EOS]


def build_corpus(sentences=None):
    """Turn a list of sentences into a tokenized corpus (list of lists)."""
    if sentences is None:
        sentences = SENTENCES
    return [tokenize(s) for s in sentences]


class NgramModel:
    """Holds unigram / bigram / trigram counts and their MLE probabilities."""

    def __init__(self, corpus=None):
        self.corpus = corpus if corpus is not None else build_corpus()

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self._count_ngrams()

        self.total_words = sum(self.unigram_counts.values())
        self.vocab = set(self.unigram_counts)

    # -- Counting (slides 5, 6, 12) ---------------------------------------
    def _count_ngrams(self):
        for sentence in self.corpus:
            self.unigram_counts.update(sentence)

            for i in range(len(sentence) - 1):
                self.bigram_counts[(sentence[i], sentence[i + 1])] += 1

            for i in range(len(sentence) - 2):
                w1, w2, w3 = sentence[i:i + 3]
                self.trigram_counts[(w1, w2, w3)] += 1

    # -- MLE probabilities (slides 5, 7, 12) ------------------------------
    def unigram_prob(self, word):
        """P(w) = C(w) / N"""
        if self.total_words == 0:
            return 0.0
        return self.unigram_counts[word] / self.total_words

    def bigram_prob(self, previous_word, word):
        """P(w_i+1 | w_i) = C(w_i, w_i+1) / C(w_i)"""
        numerator = self.bigram_counts[(previous_word, word)]
        denominator = self.unigram_counts[previous_word]
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def trigram_prob(self, w1, w2, w3):
        """P(w_i+2 | w_i, w_i+1) = C(w1, w2, w3) / C(w1, w2)

        Slide 12 writes bigram__counts (two underscores); that is a typo.
        """
        numerator = self.trigram_counts[(w1, w2, w3)]
        denominator = self.bigram_counts[(w1, w2)]
        if denominator == 0:
            return 0.0
        return numerator / denominator

    # -- Handling zero probabilities: backoff / interpolation -------------
    def backoff_prob(self, previous_word, word, alpha=0.4):
        """Stupid backoff: fall back to a discounted unigram when the bigram is 0."""
        p = self.bigram_prob(previous_word, word)
        if p > 0:
            return p
        return alpha * self.unigram_prob(word)

    def interpolation_prob(self, previous_word, word, lambda1=0.7, lambda2=0.3):
        """Linear interpolation: lambda1 * P(w|prev) + lambda2 * P(w)"""
        return (lambda1 * self.bigram_prob(previous_word, word)
                + lambda2 * self.unigram_prob(word))

    def prob_fn(self, mode="mle"):
        """Pick a probability function: 'mle' | 'backoff' | 'interpolation'."""
        if mode == "mle":
            return self.bigram_prob
        if mode == "backoff":
            return self.backoff_prob
        if mode == "interpolation":
            return self.interpolation_prob
        raise ValueError("unknown mode: " + str(mode))

    # -- Next-word prediction (slide 8) -----------------------------------
    def predict_bigram(self, previous_word, top_k=3):
        """The top_k most likely words to follow previous_word."""
        candidates = []
        for (w1, w2), _count in self.bigram_counts.items():
            if w1 == previous_word:
                candidates.append((w2, self.bigram_prob(w1, w2)))
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]

    def predict_trigram(self, w1, w2, top_k=3):
        """The top_k most likely words to follow the two-word context (w1, w2)."""
        candidates = []
        for (a, b, c), _count in self.trigram_counts.items():
            if a == w1 and b == w2:
                candidates.append((c, self.trigram_prob(a, b, c)))
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]

    # -- Sentence probability / perplexity (slide 11) ---------------------
    def sentence_log_prob(self, sentence, mode="mle"):
        """Log probability of a sentence; -inf as soon as any factor is 0."""
        prob = self.prob_fn(mode)
        tokens = tokenize(sentence)
        log_prob = 0.0
        for i in range(len(tokens) - 1):
            p = prob(tokens[i], tokens[i + 1])
            if p == 0:
                return float("-inf")
            log_prob += math.log(p)
        return log_prob

    def sentence_prob(self, sentence, mode="mle"):
        """Plain (non-log) sentence probability; 0 means the zero-probability problem."""
        log_prob = self.sentence_log_prob(sentence, mode=mode)
        return 0.0 if log_prob == float("-inf") else math.exp(log_prob)

    def perplexity(self, sentence, mode="mle"):
        """PP(W) = exp( -1/N * sum_i log P(w_i | context) )"""
        tokens = tokenize(sentence)
        n = len(tokens) - 1
        log_prob = self.sentence_log_prob(sentence, mode=mode)
        if log_prob == float("-inf"):
            return float("inf")
        return math.exp(-log_prob / n)

    # -- Sentence generation (slide 9) ------------------------------------
    def sample_next_word(self, previous_word, rng=random):
        """Sample the next word, weighting candidates by their bigram counts."""
        candidates = []
        weights = []
        for (w1, w2), count in self.bigram_counts.items():
            if w1 == previous_word:
                candidates.append(w2)
                weights.append(count)
        if not candidates:
            return EOS
        return rng.choices(candidates, weights=weights, k=1)[0]

    def generate_sentence(self, max_length=15, rng=random):
        """Start at <s> and emit words until </s> or max_length."""
        current = BOS
        output = []
        for _ in range(max_length):
            next_word = self.sample_next_word(current, rng=rng)
            if next_word == EOS:
                break
            output.append(next_word)
            current = next_word
        return " ".join(output)

    # -- HW#2: trigram-based generation (slide 13) ------------------------
    def sample_next_word_trigram(self, w1, w2, rng=random):
        """Sample on a two-word context, backing off to the bigram when unseen."""
        candidates = []
        weights = []
        for (a, b, c), count in self.trigram_counts.items():
            if a == w1 and b == w2:
                candidates.append(c)
                weights.append(count)
        if not candidates:
            return self.sample_next_word(w2, rng=rng)
        return rng.choices(candidates, weights=weights, k=1)[0]

    def generate_sentence_trigram(self, max_length=15, rng=random):
        """Generate with the trigram model; the first word comes from (<s>, w)."""
        first = self.sample_next_word(BOS, rng=rng)
        if first == EOS:
            return ""
        prev1, prev2 = BOS, first
        output = [first]
        for _ in range(max_length - 1):
            next_word = self.sample_next_word_trigram(prev1, prev2, rng=rng)
            if next_word == EOS:
                break
            output.append(next_word)
            prev1, prev2 = prev2, next_word
        return " ".join(output)
