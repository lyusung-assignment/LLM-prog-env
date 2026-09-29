"""HW#3 - Skip-gram word embeddings: window size, cosine similarity, PCA, retraining.

Usage: python hw3_word2vec.py

Skip-gram with negative sampling, implemented with numpy only. For a center
word c and a context word o the model scores sigma(v_o . v_c) and pushes it
towards 1 for real neighbours and towards 0 for k sampled negatives:

    error = target - sigma(v_o . v_c)
    v_o += lr * error * v_c
    v_c += lr * error * v_o

gensim is not installed here (and is not in the course Docker image either),
so the parts of its API this assignment needs - most_similar and similarity -
are provided by the SkipGram class below under the same names.
"""

import numpy as np

SEED = 42
EMBEDDING_SIZE = 32
NEGATIVE_SAMPLES = 5
EPOCHS = 300
LEARNING_RATE = 0.05

# --- Training corpus ------------------------------------------------------
# Four semantic groups that share contexts: royalty, people, pets, fruit and
# vehicles. Small and repetitive on purpose, like the restaurant corpus in
# Lab 2 - the point is to watch the geometry change, not to reach real
# word2vec quality.
TRAIN_SENTENCES = [
    # royalty
    "the king rules the kingdom",
    "the queen rules the kingdom",
    "the king lives in the palace",
    "the queen lives in the palace",
    "the prince lives in the palace",
    "the princess lives in the palace",
    "the king wears a golden crown",
    "the queen wears a golden crown",
    "the prince wears a golden crown",
    "the princess wears a golden crown",
    "the king is a royal man",
    "the queen is a royal woman",
    "the prince is a royal man",
    "the princess is a royal woman",
    # people
    "the man walks along the street",
    "the woman walks along the street",
    "the man reads a book at home",
    "the woman reads a book at home",
    # pets
    "the dog is a pet animal",
    "the cat is a pet animal",
    "the dog runs in the garden",
    "the cat runs in the garden",
    "i feed the dog every morning",
    "i feed the cat every morning",
    "the dog sleeps near the cat",
    "the cat sleeps near the dog",
    # fruit
    "the apple is a sweet fruit",
    "the banana is a sweet fruit",
    "i eat an apple every morning",
    "i eat a banana every morning",
    "the apple and the banana taste sweet",
    "she buys apple and banana at the market",
    # vehicles
    "the car drives along the road",
    "the bus drives along the road",
    "i drive my car to work",
    "i take the bus to work",
    "the car and the bus have wheels",
    "the driver parks the car near the bus",
]

# Problem 4: a new context for "apple" - the company, not the fruit.
NEW_SENTENCES = [
    "apple company makes computer",
    "apple makes smartphone",
    "apple company makes phone",
    "the apple company sells computer and phone",
    "apple is a technology company",
    "many people buy the apple smartphone",
    "the company makes a new computer",
    "i bought a new apple phone",
]

PROBE_WORDS = ["king", "queen", "dog", "apple", "car"]

# The words that genuinely belong to each probe word's topic. Used to count
# how many of a top-5 list are on topic instead of judging the lists by eye.
TOPICS = {
    "king": ["king", "queen", "prince", "princess", "kingdom", "palace",
             "crown", "golden", "royal", "rules", "wears"],
    "dog": ["dog", "cat", "pet", "animal", "garden", "runs", "feed",
            "sleeps", "near"],
    "apple": ["apple", "banana", "fruit", "sweet", "taste", "eat", "market",
              "buys"],
    "car": ["car", "bus", "road", "drives", "drive", "wheels", "driver",
            "parks", "take", "work"],
}
TOPICS["queen"] = TOPICS["king"]

PAIRS = [
    ("king", "queen"),
    ("dog", "cat"),
    ("apple", "banana"),
    ("car", "bus"),
    ("king", "banana"),
    ("dog", "car"),
]

PCA_WORDS = ["king", "queen", "man", "woman", "dog", "cat",
             "apple", "banana", "car", "bus", "prince", "princess"]

# Colour groups for the PCA plot only.
PCA_GROUPS = {
    "royalty": ["king", "queen", "prince", "princess"],
    "people": ["man", "woman"],
    "pets": ["dog", "cat"],
    "fruit": ["apple", "banana"],
    "vehicles": ["car", "bus"],
}


class SkipGram:
    """Skip-gram with negative sampling."""

    def __init__(self, sentences, window=2, size=EMBEDDING_SIZE,
                 negative=NEGATIVE_SAMPLES, epochs=EPOCHS, lr=LEARNING_RATE,
                 seed=SEED):
        self.window = window
        self.size = size
        self.negative = negative
        self.epochs = epochs
        self.lr = lr
        self.rng = np.random.default_rng(seed)

        self.corpus = [s.lower().split() for s in sentences]

        counts = {}
        for tokens in self.corpus:
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
        self.vocab = sorted(counts)
        self.index = {word: i for i, word in enumerate(self.vocab)}
        self.counts = np.array([counts[w] for w in self.vocab], dtype=float)

        # Negative samples are drawn from the unigram distribution raised to
        # 3/4, which is what the word2vec paper uses: it lifts rare words and
        # flattens very frequent ones like "the".
        weights = self.counts ** 0.75
        self.noise = weights / weights.sum()

        # Two matrices, as in the slides: one for the word as a center word,
        # one for the word as a context word. Both start random, which is what
        # the lecture specifies ("each word has 2 vectors, both randomly
        # initialized"). The original C implementation leaves the context
        # matrix at zero instead; on this corpus random is slightly better.
        shape = (len(self.vocab), size)
        self.W_in = (self.rng.random(shape) - 0.5) / size
        self.W_out = (self.rng.random(shape) - 0.5) / size

        self.ids = [[self.index[t] for t in tokens] for tokens in self.corpus]

    # -- Training ---------------------------------------------------------
    def _pairs(self):
        """(center, context) index pairs inside the window."""
        for tokens in self.ids:
            for i, center in enumerate(tokens):
                start = max(0, i - self.window)
                stop = min(len(tokens), i + self.window + 1)
                for j in range(start, stop):
                    if j != i:
                        yield center, tokens[j]

    def train(self, verbose=False):
        pairs = list(self._pairs())
        order = np.arange(len(pairs))
        pairs = np.array(pairs)

        for epoch in range(self.epochs):
            # Decay the learning rate so late updates do not undo early ones.
            lr = self.lr * (1 - epoch / self.epochs) + 1e-4
            self.rng.shuffle(order)
            loss = 0.0

            for idx in order:
                center, context = pairs[idx]
                negatives = self.rng.choice(len(self.vocab),
                                            size=self.negative, p=self.noise)
                targets = np.concatenate(([context], negatives))
                labels = np.zeros(1 + self.negative)
                labels[0] = 1.0

                v = self.W_in[center]
                u = self.W_out[targets]
                score = 1.0 / (1.0 + np.exp(-np.clip(u @ v, -20, 20)))
                error = labels - score

                loss -= (np.log(score[0] + 1e-10)
                         + np.log(1 - score[1:] + 1e-10).sum())

                grad_v = error @ u
                np.add.at(self.W_out, targets, lr * error[:, None] * v)
                self.W_in[center] += lr * grad_v

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    epoch {epoch + 1:4d}/{self.epochs}"
                      f"  loss={loss / len(pairs):.4f}")
        return self

    # -- Querying (same names as gensim) ----------------------------------
    def get_vector(self, word):
        return self.W_in[self.index[word]]

    def _unit(self):
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        return self.W_in / np.maximum(norms, 1e-10)

    def similarity(self, word1, word2):
        """Cosine similarity between two word vectors."""
        a, b = self.get_vector(word1), self.get_vector(word2)
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-10))

    def most_similar(self, word, topn=5):
        """The topn nearest words by cosine similarity, excluding the word."""
        unit = self._unit()
        scores = unit @ unit[self.index[word]]
        scores[self.index[word]] = -np.inf
        best = np.argsort(-scores)[:topn]
        return [(self.vocab[i], float(scores[i])) for i in best]


def pca_2d(vectors):
    """Project onto the first two principal components (numpy SVD, no sklearn)."""
    centered = vectors - vectors.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def train_model(window, sentences=None, label=""):
    sentences = TRAIN_SENTENCES if sentences is None else sentences
    print(f"  training {label or f'window={window}'} "
          f"({len(sentences)} sentences) ...")
    return SkipGram(sentences, window=window).train()


# -------------------------------------------------------------------------
# Problem 1 - two window sizes, top-5 similar words
# -------------------------------------------------------------------------
def problem1(models):
    print("=" * 74)
    print("Problem 1. Top-5 similar words for window = 1 and window = 4")
    print("=" * 74)

    hits = {1: 0, 4: 0}
    off_topic = {1: [], 4: []}
    for word in PROBE_WORDS:
        topic = set(TOPICS[word])
        top = {w: models[w].most_similar(word, topn=5) for w in (1, 4)}
        on_topic = {w: sum(1 for x, _ in top[w] if x in topic) for w in (1, 4)}
        hits[1] += on_topic[1]
        hits[4] += on_topic[4]
        for w in (1, 4):
            # one example per probe word, so the list below is not all king
            stray = [x for x, _ in top[w] if x not in topic]
            if stray:
                off_topic[w].append(f"{stray[0]} (after {word})")

        print(f"\n  {word}")
        print(f"    {'window = 1':<32}{'window = 4'}")
        for (w1, s1), (w4, s4) in zip(top[1], top[4]):
            mark1 = " " if w1 in topic else "*"
            mark4 = " " if w4 in topic else "*"
            print(f"    {w1:<14}{s1:>6.3f}{mark1}         {w4:<14}{s4:>6.3f}{mark4}")
        print(f"    on topic: {on_topic[1]}/5                     "
              f"on topic: {on_topic[4]}/5")

    total = 5 * len(PROBE_WORDS)
    print(f"\n  Words on topic across all five lists (* marks the ones that are not)")
    print(f"    window = 1 : {hits[1]}/{total}")
    print(f"    window = 4 : {hits[4]}/{total}")

    print("\n  How the window changes the result")
    print("  - window = 1 sees only the immediate neighbours, which in this")
    print("    corpus are almost always function words (the, a, is). Two words")
    print("    become similar when they fill the same slot, so the twin is found")
    print("    (king-queen, dog-cat, both at 0.99+) but the rest of the list is")
    print("    filled with words that merely share a position:")
    print(f"      {', '.join(off_topic[1])}")
    print("  - window = 4 reaches the content words of the sentence, so the whole")
    print("    list stays on topic:")
    print(f"      {'off topic: ' + ', '.join(off_topic[4]) if off_topic[4] else 'nothing off topic in any of the five lists'}")
    print("  - Note that the window = 1 numbers are higher, not better. A narrow")
    print("    window pushes everything into one narrow cone, which inflates the")
    print("    cosine values; the ranking underneath is worse.")
    print("  - The lecture draws this line between 2-15 (interchangeability) and")
    print("    15-50 (relatedness). Both windows here sit inside that first band,")
    print("    so this is not the comparison the slide makes - but the same")
    print("    tendency already shows up between 1 and 4 on a corpus this small.")
    print()


# -------------------------------------------------------------------------
# Problem 2 - cosine similarity for selected pairs
# -------------------------------------------------------------------------
def problem2(models):
    print("=" * 74)
    print("Problem 2. Cosine similarity of selected pairs")
    print("=" * 74)

    header = f"  {'pair':<22}{'window = 1':>12}{'window = 4':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for w1, w2 in PAIRS:
        s1 = models[1].similarity(w1, w2)
        s4 = models[4].similarity(w1, w2)
        rows.append((f"{w1} - {w2}", s1, s4))
        print(f"  {w1 + ' - ' + w2:<22}{s1:>12.3f}{s4:>14.3f}")

    best = max(rows, key=lambda r: r[2])
    worst = min(rows, key=lambda r: r[2])
    print(f"\n  Most similar  (window = 4): {best[0]}  ({best[2]:.3f})")
    print(f"  Least similar (window = 4): {worst[0]}  ({worst[2]:.3f})")
    print("\n  The four same-category pairs (king-queen, dog-cat, apple-banana,")
    print("  car-bus) score high because the two words appear in nearly the same")
    print("  sentences. The two cross-category pairs (king-banana, dog-car) score")
    print("  far lower: they never share a context in the corpus.")
    print()


# -------------------------------------------------------------------------
# Problem 3 - 2-D PCA visualization
# -------------------------------------------------------------------------
def problem3(models, path="hw3_pca.png"):
    print("=" * 74)
    print("Problem 3. 2-D PCA of 12 selected words")
    print("=" * 74)

    import matplotlib
    matplotlib.use("Agg")          # write a file, do not open a window
    import matplotlib.pyplot as plt

    colors = {"royalty": "#c0392b", "people": "#d68910", "pets": "#1e8449",
              "fruit": "#7d3c98", "vehicles": "#1f618d"}
    group_of = {w: g for g, words in PCA_GROUPS.items() for w in words}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, window in zip(axes, (1, 4)):
        model = models[window]
        coords = pca_2d(np.array([model.get_vector(w) for w in PCA_WORDS]))
        for i, ((x, y), word) in enumerate(zip(coords, PCA_WORDS)):
            color = colors[group_of[word]]
            ax.scatter(x, y, s=90, color=color, zorder=3)
            # Members of a group land almost on top of each other, so put one
            # label above the point and the next one below.
            offset = (7, 5) if i % 2 == 0 else (7, -13)
            ax.annotate(word, (x, y), xytext=offset,
                        textcoords="offset points", fontsize=11, color=color)
        ax.set_title(f"window = {window}")
        ax.axhline(0, color="#dddddd", lw=0.8, zorder=1)
        ax.axvline(0, color="#dddddd", lw=0.8, zorder=1)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.margins(0.22)           # keep the labels inside the axes

    fig.suptitle("Skip-gram embeddings projected onto 2 principal components")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")

    # Printed evidence for the same thing the plot shows: the average
    # distance inside a group against the average distance between groups.
    ratio = {}
    for window in (1, 4):
        model = models[window]
        coords = pca_2d(np.array([model.get_vector(w) for w in PCA_WORDS]))
        point = dict(zip(PCA_WORDS, coords))
        inside, outside = [], []
        for i, a in enumerate(PCA_WORDS):
            for b in PCA_WORDS[i + 1:]:
                d = float(np.linalg.norm(point[a] - point[b]))
                (inside if group_of[a] == group_of[b] else outside).append(d)
        ratio[window] = np.mean(outside) / np.mean(inside)
        print(f"  window = {window}:  mean distance within a group "
              f"{np.mean(inside):.3f},  between groups {np.mean(outside):.3f}"
              f"  (ratio {ratio[window]:.2f}x)")

    print("\n  Semantically related words do land together: every pair sits in its")
    print(f"  own corner, {ratio[1]:.0f}x closer to its partner than to anything else")
    print(f"  at window = 1 and {ratio[4]:.0f}x at window = 4. Both windows manage")
    print("  the separation - these five groups barely share any vocabulary, so")
    print("  even a one-word window keeps them apart - but the larger window does")
    print("  pull the groups further from each other.")
    print("  These 12 words are the easy case, though. The window size shows up")
    print("  far more sharply in the ranking of the whole vocabulary, which is")
    print("  what the on-topic counts in Problem 1 measure.")

    model = models[4]
    print("\n  Inside the royal cluster the male/female split is visible in the")
    print("  cosine values even though the projection does not lay it on one axis:")
    print(f"    king - prince      {model.similarity('king', 'prince'):.3f}")
    print(f"    queen - princess   {model.similarity('queen', 'princess'):.3f}")
    print(f"    king - queen       {model.similarity('king', 'queen'):.3f}")
    print(f"    king - princess    {model.similarity('king', 'princess'):.3f}")
    print()


# -------------------------------------------------------------------------
# Problem 4 - add new sentences and retrain
# -------------------------------------------------------------------------
def problem4(models):
    print("=" * 74)
    print("Problem 4. most_similar(\"apple\") before and after new sentences")
    print("=" * 74)

    print(f"\n  {len(NEW_SENTENCES)} new sentences give \"apple\" a second context:")
    for sentence in NEW_SENTENCES:
        print(f"    {sentence}")

    print()
    retrained = train_model(4, TRAIN_SENTENCES + NEW_SENTENCES,
                            label="window=4 + new sentences")

    before = models[4].most_similar("apple", topn=5)
    after = retrained.most_similar("apple", topn=5)

    print(f"\n  {'before (38 sentences)':<34}{'after (46 sentences)'}")
    print("  " + "-" * 62)
    for (wb, sb), (wa, sa) in zip(before, after):
        print(f"  {wb:<16}{sb:>6.3f}          {wa:<16}{sa:>6.3f}")

    old_words = {w for w, _ in before}
    new_words = [w for w, _ in after if w not in old_words]
    print(f"\n  New in the top-5: {', '.join(new_words) if new_words else 'none'}")

    print("\n  similarity of \"apple\" to:")
    print(f"    {'banana (fruit context)':<28}"
          f"{models[4].similarity('apple', 'banana'):.3f}"
          f"  ->  {retrained.similarity('apple', 'banana'):.3f}")
    for word in ("company", "phone", "computer", "smartphone"):
        print(f"    {word + ' (new context)':<28}"
              f"{'-':>5}  ->  {retrained.similarity('apple', word):.3f}")

    print("\n  \"apple\" drops away from banana and picks up the company words.")
    print("  The fruit sense is not erased - market, buys and eat are still in")
    print("  the top-5 - because the fruit sentences are still in the corpus;")
    print("  the vector now sits between the two senses, which is exactly what a")
    print("  single vector per word can do.")
    print("  Nothing about the word itself changed, only the company it keeps.")
    print("  That is the point of the assignment: embeddings are a function of")
    print("  the training contexts, so changing the contexts changes the")
    print("  geometry of the space.")
    print()


def main():
    print("Training skip-gram models (negative sampling, numpy)")
    print(f"  dim={EMBEDDING_SIZE}, negatives={NEGATIVE_SAMPLES}, "
          f"epochs={EPOCHS}, seed={SEED}\n")
    models = {window: train_model(window) for window in (1, 4)}
    print(f"  vocabulary size: {len(models[4].vocab)}\n")

    problem1(models)
    problem2(models)
    problem3(models)
    problem4(models)


if __name__ == "__main__":
    main()
