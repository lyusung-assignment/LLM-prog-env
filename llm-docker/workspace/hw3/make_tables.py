"""Write the tables for the HW#3 report to TABLES.md (markdown, paste-ready).

Usage: python make_tables.py

Not part of the assignment itself - it just re-reads the numbers out of the
models in hw3_word2vec.py so the report never disagrees with the code.
"""

import io
import os

import numpy as np

import hw3_word2vec as hw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TABLES.md")


def table(headers, rows):
    line = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([line, rule] + body)


def main():
    print("training models ...")
    models = {w: hw.SkipGram(hw.TRAIN_SENTENCES, window=w).train() for w in (1, 4)}
    retrained = hw.SkipGram(hw.TRAIN_SENTENCES + hw.NEW_SENTENCES,
                            window=4).train()

    out = []
    add = out.append

    add("# HW#3 보고서용 표\n")
    add("`python make_tables.py` 로 생성. 숫자는 모델에서 직접 읽은 값이다.\n")

    # --- setup ----------------------------------------------------------
    add("## 표 0. 실험 설정\n")
    add(table(["항목", "값"], [
        ["모델", "Skip-gram + negative sampling (numpy 직접 구현)"],
        ["훈련 문장", f"{len(hw.TRAIN_SENTENCES)}문장 (문제 4에서 "
                      f"{len(hw.NEW_SENTENCES)}문장 추가)"],
        ["어휘 크기", len(models[4].vocab)],
        ["임베딩 차원", hw.EMBEDDING_SIZE],
        ["negative sample 수", f"{hw.NEGATIVE_SAMPLES} (슬라이드 p.59 권장 5~20)"],
        ["epochs / learning rate", f"{hw.EPOCHS} / {hw.LEARNING_RATE} (선형 감쇠)"],
        ["random seed", hw.SEED],
        ["초기화", "embedding·context 행렬 모두 랜덤 (슬라이드 p.45)"],
    ]))
    add("")

    # --- problem 1 ------------------------------------------------------
    add("## 표 1. 문제 1 - window 크기별 top-5 유사 단어\n")
    rows = []
    hits = {1: 0, 4: 0}
    for word in hw.PROBE_WORDS:
        topic = set(hw.TOPICS[word])
        top = {w: models[w].most_similar(word, topn=5) for w in (1, 4)}
        for rank in range(5):
            w1, s1 = top[1][rank]
            w4, s4 = top[4][rank]
            rows.append([word if rank == 0 else "", rank + 1,
                         w1 + ("*" if w1 not in topic else ""), f"{s1:.3f}",
                         w4 + ("*" if w4 not in topic else ""), f"{s4:.3f}"])
        for w in (1, 4):
            hits[w] += sum(1 for x, _ in top[w] if x in topic)
    add(table(["기준 단어", "순위", "window=1", "유사도", "window=4", "유사도"], rows))
    add("\n`*` 는 해당 단어의 주제어 목록(코드의 `TOPICS`)에 없는 단어.\n")

    add("## 표 2. 문제 1 - top-5 안의 주제 적중 수\n")
    total = 5 * len(hw.PROBE_WORDS)
    per_word = []
    for word in hw.PROBE_WORDS:
        topic = set(hw.TOPICS[word])
        per_word.append([word] + [
            f"{sum(1 for x, _ in models[w].most_similar(word, 5) if x in topic)} / 5"
            for w in (1, 4)])
    per_word.append(["**합계**", f"**{hits[1]} / {total}**", f"**{hits[4]} / {total}**"])
    add(table(["기준 단어", "window=1", "window=4"], per_word))
    add("")

    # --- problem 2 ------------------------------------------------------
    add("## 표 3. 문제 2 - 선택한 쌍의 cosine similarity\n")
    # 범주는 코퍼스를 만들 때 정한 그룹에서 가져온다. 유사도로 판정하면
    # 보여주려는 수치로 정답을 되짚는 꼴이 된다.
    group_of = {w: g for g, words in hw.PCA_GROUPS.items() for w in words}
    rows = []
    for a, b in hw.PAIRS:
        s1, s4 = models[1].similarity(a, b), models[4].similarity(a, b)
        same = "같은 범주" if group_of[a] == group_of[b] else "다른 범주"
        rows.append([f"{a} - {b}", f"{s1:.3f}", f"{s4:.3f}", same])
    add(table(["쌍", "window=1", "window=4", "비고"], rows))
    scores = {f"{a} - {b}": models[4].similarity(a, b) for a, b in hw.PAIRS}
    top = max(scores, key=scores.get)
    bottom = min(scores, key=scores.get)
    add(f"\nwindow=4 기준 최고: **{top} ({scores[top]:.3f})**, "
        f"최저: **{bottom} ({scores[bottom]:.3f})**\n")

    # --- problem 3 ------------------------------------------------------
    add("## 표 4. 문제 3 - PCA 평면에서의 그룹 분리도\n")
    rows = []
    for window in (1, 4):
        coords = hw.pca_2d(np.array([models[window].get_vector(w)
                                     for w in hw.PCA_WORDS]))
        point = dict(zip(hw.PCA_WORDS, coords))
        inside, outside = [], []
        for i, a in enumerate(hw.PCA_WORDS):
            for b in hw.PCA_WORDS[i + 1:]:
                d = float(np.linalg.norm(point[a] - point[b]))
                (inside if group_of[a] == group_of[b] else outside).append(d)
        rows.append([f"window={window}", f"{np.mean(inside):.3f}",
                     f"{np.mean(outside):.3f}",
                     f"{np.mean(outside) / np.mean(inside):.2f}x"])
    add(table(["", "그룹 내 평균거리", "그룹 간 평균거리", "비율"], rows))
    add("\n그림: `hw3_pca.png` (두 윈도우를 나란히 그린 것). PCA 축의 부호는 "
        "SVD 관례상 임의이므로 두 그림의 축 방향을 서로 비교하면 안 된다.\n")

    add("## 표 5. 문제 3 - 왕족 클러스터 내부의 남/여 구분 (window=4)\n")
    add(table(["쌍", "cosine"], [
        [f"{a} - {b}", f"{models[4].similarity(a, b):.3f}"]
        for a, b in [("king", "prince"), ("queen", "princess"),
                     ("king", "queen"), ("king", "princess")]]))
    add("\n2-D 투영에서는 한 축으로 정렬되지 않지만 cosine 값에는 남아 있다.\n")

    # --- problem 4 ------------------------------------------------------
    add("## 표 6. 문제 4 - 추가한 훈련 문장\n")
    add(table(["#", "문장"],
              [[i, s] for i, s in enumerate(hw.NEW_SENTENCES, start=1)]))
    add("")

    add("## 표 7. 문제 4 - `most_similar(\"apple\")` 전후 비교\n")
    before = models[4].most_similar("apple", topn=5)
    after = retrained.most_similar("apple", topn=5)
    add(table(["순위",
               f"before ({len(hw.TRAIN_SENTENCES)}문장)", "유사도",
               f"after ({len(hw.TRAIN_SENTENCES) + len(hw.NEW_SENTENCES)}문장)",
               "유사도"],
              [[i + 1, wb, f"{sb:.3f}", wa, f"{sa:.3f}"]
               for i, ((wb, sb), (wa, sa)) in enumerate(zip(before, after))]))
    add("")

    add("## 표 8. 문제 4 - `apple` 과의 유사도 변화\n")
    rows = [["banana", "과일 문맥", f"{models[4].similarity('apple', 'banana'):.3f}",
             f"{retrained.similarity('apple', 'banana'):.3f}"]]
    for word in ("phone", "computer", "company", "smartphone"):
        rows.append([word, "새 문맥", "어휘에 없음",
                     f"{retrained.similarity('apple', word):.3f}"])
    add(table(["상대 단어", "구분", "before", "after"], rows))
    add("")

    io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
