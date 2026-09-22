"""HW#2 - trigram sentence generation and unigram/bigram/trigram perplexity.

Usage: python hw2_ngram.py
"""

import math
import random
from collections import Counter

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

TRAIN_SENTENCES = [
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

# None of these appear in TRAIN_SENTENCES. The last one has an unknown word.
TEST_SENTENCES = [
    "i want to eat italian food",
    "i want a chinese restaurant",
    "can you find italian food",
    "tell me about italian restaurants",
    "i am looking for a thai restaurant",
    "where can i eat thai food",
    "i am looking for italian food",
    "where can i find a korean restaurant",
]


class NgramLM:
    """Add-k smoothed n-gram model: P(w|ctx) = (C(ctx,w) + k) / (C(ctx) + kV)."""

    def __init__(self, order, sentences, k=1.0):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.k = k
        self.sentences = list(sentences)

        # The vocabulary is what the model has to predict: the words and </s>,
        # never <s>, plus one slot for unknown test words.
        self.vocab = {UNK, EOS}
        for sentence in self.sentences:
            self.vocab.update(sentence.lower().split())
        self.vocab_size = len(self.vocab)

        self.ngram_counts = Counter()
        self.context_counts = Counter()
        for sentence in self.sentences:
            for context, word in self._ngrams(self.tokenize(sentence)):
                self.ngram_counts[(context, word)] += 1
                self.context_counts[context] += 1

    def tokenize(self, sentence):
        words = [w if w in self.vocab else UNK for w in sentence.lower().split()]
        return [BOS] * (self.order - 1) + words + [EOS]

    def _ngrams(self, tokens):
        """(context, word) pairs; the context is empty for a unigram."""
        for i in range(self.order - 1, len(tokens)):
            yield tuple(tokens[i - self.order + 1:i]), tokens[i]

    def prob(self, context, word):
        numerator = self.ngram_counts[(context, word)] + self.k
        denominator = self.context_counts[context] + self.k * self.vocab_size
        return numerator / denominator

    def sentence_log_prob(self, sentence):
        """Log probability and the number of tokens it was computed over."""
        log_prob = 0.0
        count = 0
        for context, word in self._ngrams(self.tokenize(sentence)):
            log_prob += math.log(self.prob(context, word))
            count += 1
        return log_prob, count

    def perplexity(self, sentence):
        log_prob, count = self.sentence_log_prob(sentence)
        return math.exp(-log_prob / count)

    def corpus_perplexity(self, sentences):
        total_log_prob = 0.0
        total_count = 0
        for sentence in sentences:
            log_prob, count = self.sentence_log_prob(sentence)
            total_log_prob += log_prob
            total_count += count
        return math.exp(-total_log_prob / total_count)

    def candidates(self, context):
        """Every (word, count) observed after this context."""
        return [(word, count)
                for (ctx, word), count in self.ngram_counts.items()
                if ctx == context]


def generate_sentence(models, max_length=15, rng=random):
    """Generate a sentence with the trigram model, backing off when needed.

    Sampling uses the raw counts, not the smoothed distribution: smoothing
    puts a little probability on the whole vocabulary, which is right for
    scoring unseen text but would drop random words into the output.
    """
    history = [BOS, BOS]
    output = []
    for _ in range(max_length):
        for order in (3, 2, 1):
            context = tuple(history[len(history) - (order - 1):]) if order > 1 else ()
            candidates = models[order].candidates(context)
            if candidates:
                break
        words = [w for w, _ in candidates]
        weights = [c for _, c in candidates]
        next_word = rng.choices(words, weights=weights, k=1)[0]
        if next_word == EOS:
            break
        output.append(next_word)
        history.append(next_word)
    return " ".join(output)


def problem1(models, n_sentences=5):
    print("=" * 68)
    print("Problem 1. Trigram model - sentence generation")
    print("=" * 68)

    trigram = models[3]
    print(f"training sentences : {len(trigram.sentences)}")
    print(f"vocabulary size V  : {trigram.vocab_size}")
    print(f"distinct trigrams  : {len(trigram.ngram_counts)}")

    context = ("i", "want")
    total = trigram.context_counts[context]
    print("\nP(w | i, want):")
    for word, count in sorted(trigram.candidates(context), key=lambda x: -x[1]):
        print(f"  {word:10s} {count}/{total} = {count / total:.3f}")

    print(f"\n{n_sentences} generated sentences:")
    generated = [generate_sentence(models) for _ in range(n_sentences)]
    train_set = {s.lower() for s in trigram.sentences}
    for i, sentence in enumerate(generated, start=1):
        tag = "(seen in training)" if sentence in train_set else "(new)"
        print(f"  {i}. {sentence:38s} {tag}")

    memorized = sum(1 for s in generated if s in train_set)
    print(f"\n{memorized}/{n_sentences} are copies of a training sentence. With 15 sentences")
    print("a two-word context almost always has one continuation, so the model")
    print("mostly replays what it saw.")
    print()


def problem2(models, test_sentences):
    print("=" * 68)
    print("Problem 2. Perplexity on held-out sentences")
    print("=" * 68)

    train_set = {s.lower() for s in models[1].sentences}
    overlap = [s for s in test_sentences if s.lower() in train_set]
    assert not overlap, f"test sentence also used in training: {overlap}"
    print(f"test sentences: {len(test_sentences)}, none used in training")
    print(f"smoothing     : add-k, k = {models[1].k}")

    header = f"{'test sentence':38s}{'unigram':>10s}{'bigram':>10s}{'trigram':>10s}"
    print()
    print(header)
    print("-" * len(header))
    for sentence in test_sentences:
        row = "".join(f"{models[order].perplexity(sentence):>10.2f}"
                      for order in (1, 2, 3))
        print(f"{sentence:38s}{row}")

    print("-" * len(header))
    corpus = {order: models[order].corpus_perplexity(test_sentences)
              for order in (1, 2, 3)}
    print(f"{'corpus perplexity':38s}"
          + "".join(f"{corpus[order]:>10.2f}" for order in (1, 2, 3)))

    train_pp = {order: models[order].corpus_perplexity(models[1].sentences)
                for order in (1, 2, 3)}
    print(f"{'  (on the training sentences)':38s}"
          + "".join(f"{train_pp[order]:>10.2f}" for order in (1, 2, 3)))

    names = {1: "unigram", 2: "bigram", 3: "trigram"}
    best = min(corpus, key=corpus.get)
    print(f"\nBest on the test set: {names[best]}, PP = {corpus[best]:.2f}")
    print("Unigram ignores the context entirely, so it is the worst. Trigram wins")
    print("on the training sentences but loses on the test set: its contexts are")
    print("too sparse, and an unseen context falls back to the flat 1/V estimate.")
    print()


def smoothing_sweep(test_sentences, ks=(1.0, 0.1, 0.01, 0.001)):
    """Check that the ranking above is not an artifact of the choice of k."""
    print("=" * 68)
    print("Test perplexity vs. the smoothing constant k")
    print("=" * 68)
    print(f"{'k':>8s}{'unigram':>12s}{'bigram':>12s}{'trigram':>12s}")
    print("-" * 44)
    for k in ks:
        row = ""
        for order in (1, 2, 3):
            model = NgramLM(order, TRAIN_SENTENCES, k=k)
            row += f"{model.corpus_perplexity(test_sentences):>12.2f}"
        print(f"{k:>8g}{row}")
    print("\nBigram stays ahead of trigram at every k, so the ranking comes from")
    print("the size of the corpus, not from the smoothing constant.")
    print()


def main():
    random.seed(42)
    models = {order: NgramLM(order, TRAIN_SENTENCES, k=1.0) for order in (1, 2, 3)}

    problem1(models)
    problem2(models, TEST_SENTENCES)
    smoothing_sweep(TEST_SENTENCES)


if __name__ == "__main__":
    main()
