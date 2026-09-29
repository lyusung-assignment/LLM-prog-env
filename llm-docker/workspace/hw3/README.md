# HW#3 - Skip-gram Word Embeddings

`HW#3.pdf` 과제 (LLM Programming, 상명대학교 컴퓨터과학과 강상욱 교수).
강의자료는 `Lecture 2 (Ngram_Word2Vec).pdf` 의 Word2Vec 파트(p.36~64).

| 문제 | 내용 | 코드 |
| --- | --- | --- |
| 1 | window=1 / window=4 로 학습 후 top-5 유사 단어 비교 | `problem1()` |
| 2 | 지정된 6쌍의 cosine similarity, 최고/최저 식별 | `problem2()` |
| 3 | 12개 단어의 2-D PCA 시각화 | `problem3()` |
| 4 | 새 문장 8개 추가 후 재학습, `most_similar("apple")` 비교 | `problem4()` |

## 실행

```bash
python hw3_word2vec.py
```

약 30초 걸린다(모델 3개 학습). `hw3_pca.png` 가 생성된다.

## 의존성 - gensim을 쓰지 않은 이유

**이 환경에 gensim이 없다.** 로컬 Python이 3.14라 gensim 휠이 없고, 저장소의
Docker 이미지에도 들어있지 않다(`llm-docker/Dockerfile` 의 pip 목록은
scipy·scikit-learn까지). 그래서 강의자료 p.45~59 의 skip-gram + negative
sampling 을 numpy로 직접 구현했다.

- `SkipGram.most_similar()`, `SkipGram.similarity()` 는 gensim과 같은 이름으로
  맞췄으므로, gensim을 쓰는 코드와 호출부가 동일하다.
- PCA도 sklearn 없이 numpy SVD로 계산한다(`pca_2d()`).
- 따라서 **numpy와 matplotlib만 있으면 돌아간다.**

학습 규칙은 슬라이드 그대로다.

```
error = target - sigmoid(v_o · v_c)
v_o  += lr * error * v_c
v_c  += lr * error * v_o
```

negative sample은 unigram 분포의 3/4 제곱에서 뽑는다(word2vec 논문 방식.
슬라이드 p.53은 "pick randomly from vocabulary"까지만 말한다). embedding 행렬과
context 행렬 **둘 다 랜덤 초기화**하는데, 이는 슬라이드 p.45의 "each word has 2
vectors, both randomly initialized"를 따른 것이다. 원논문 C 구현은 context
행렬을 0으로 두지만, 이 코퍼스에서는 랜덤 쪽 결과가 조금 더 깨끗했다.

하이퍼파라미터: `dim=32, negatives=5, epochs=300, lr=0.05, seed=42`.
negative sample 개수 5는 슬라이드 p.59 권장 범위(5~20)의 하한이다.

## 코퍼스

38문장, 5개 의미 그룹(royalty / people / pets / fruit / vehicles)이 서로
비슷한 문맥을 공유하도록 구성했다. Lab 2의 restaurant 코퍼스처럼 작고
반복적인데, 목적이 임베딩 품질이 아니라 **문맥이 바뀌면 공간이 어떻게
바뀌는지** 보는 것이기 때문이다. 어휘는 61단어.

## 결과 요약

### 1. window=1 vs window=4

top-5 안에 해당 주제어가 몇 개 들어왔는지로 셌다(코드의 `TOPICS`).

| | 주제에 맞는 단어 |
| --- | --- |
| window = 1 | 12 / 25 |
| window = 4 | **25 / 25** |

- window=1 은 바로 옆 단어(대부분 `the`, `a`, `is`)만 보기 때문에 **같은 자리에
  들어가는 단어**가 가까워진다. 쌍둥이(king-queen 0.999, dog-cat 0.999)는 잘
  찾지만 그 뒤는 자리만 공유하는 단어로 채워진다 — king 뒤의 `market`,
  dog 뒤의 `morning`, apple 뒤의 `she`, car 뒤의 `sleeps`.
- window=4 는 문장의 내용어(`kingdom`, `pet`, `wheels`, `sweet`)까지 닿아서
  다섯 목록 **전부**가 주제 안에 머문다.
- **window=1 의 수치가 대체로 더 높지만 더 좋은 것은 아니다.** 좁은 윈도우는
  모든 벡터를 좁은 원뿔 안으로 몰아넣어 cosine 값 자체를 부풀린다(교차 범주
  쌍도 0.39 vs 0.21로 같이 올라간다). 순위는 오히려 나쁘다.
- **주의:** 슬라이드 p.59가 나누는 기준은 **2~15(interchangeability) 대
  15~50(relatedness)** 이다. 이 과제의 1과 4는 둘 다 앞 구간에 들어가므로
  슬라이드가 대조하는 두 경우가 아니다. 다만 같은 방향의 경향이 이 작은
  코퍼스에서는 1과 4 사이에서도 이미 나타난다.

### 2. Cosine similarity

| 쌍 | window=1 | window=4 |
| --- | --- | --- |
| king - queen | 0.999 | **0.832 (최고)** |
| dog - cat | 0.999 | 0.753 |
| apple - banana | 0.710 | 0.714 |
| car - bus | 0.740 | 0.565 |
| king - banana | 0.397 | 0.212 |
| dog - car | 0.385 | **0.220** |

window=4 기준 최고는 king-queen(0.832), 최저는 king-banana(0.212). 같은 범주
4쌍은 거의 같은 문장에 등장하므로 높고, 교차 범주 2쌍은 코퍼스에서 문맥을 한
번도 공유하지 않아 낮다.

### 3. PCA

`hw3_pca.png` — 두 윈도우를 나란히 그렸다. 5개 그룹이 각자의 영역에 모인다.

| | 그룹 내 평균거리 | 그룹 간 평균거리 | 비율 |
| --- | --- | --- | --- |
| window=1 | 0.232 | 2.718 | 11.71x |
| window=4 | 0.174 | 2.971 | **17.07x** |

의미가 가까운 단어들이 실제로 붙어 있고, 큰 윈도우 쪽이 그룹을 더 멀리
떼어놓는다. 다만 **두 윈도우 모두 분리에는 성공한다** — 5개 그룹이 어휘를 거의
공유하지 않아서 한 단어짜리 윈도우로도 갈라지기 때문이다. 고른 12단어는 쉬운
경우이고, 윈도우 크기의 차이는 어휘 전체의 순위(문제 1의 12/25 대 25/25)에서
훨씬 뚜렷하다.

왕족 클러스터 안의 남/여 구분은 2-D 투영에서는 한 축으로 정렬되지 않지만
cosine 값에는 남아 있다: king-prince 0.907, queen-princess 0.910 >
king-queen 0.832 > king-princess 0.740.

PCA 축의 부호는 SVD 관례상 임의로 정해지므로, 두 그림의 **축 방향을 서로
비교하면 안 된다.** 의미가 있는 것은 각 그림 안에서의 상대적 거리뿐이다.

### 4. 문장 추가 전후의 `most_similar("apple")`

`apple` 에 회사 문맥을 주는 문장 8개를 추가했다.

| 순위 | before (38문장) | after (46문장) |
| --- | --- | --- |
| 1 | banana 0.714 | she 0.489 |
| 2 | market 0.644 | market 0.487 |
| 3 | sweet 0.624 | buys 0.482 |
| 4 | buys 0.614 | **phone 0.477** |
| 5 | eat 0.596 | eat 0.472 |

`apple` 과의 유사도 변화:

| 상대 단어 | before | after |
| --- | --- | --- |
| banana (과일 문맥) | 0.714 | **0.434** |
| phone | 어휘에 없음 | 0.477 |
| computer | 어휘에 없음 | 0.469 |
| company | 어휘에 없음 | 0.426 |
| smartphone | 어휘에 없음 | 0.359 |

banana에서 멀어지고 회사 단어들이 그 자리를 채운다. 과일 의미가 지워지지는
않는데(`market`, `buys`, `eat` 이 여전히 top-5에 있다) 과일 문장이 코퍼스에
그대로 남아 있기 때문이다. 벡터가 두 의미 사이 어딘가에 놓인 것이고, 단어당
벡터가 하나뿐인 모델이 할 수 있는 일이 딱 여기까지다.

단어 자체는 그대로인데 함께 나타나는 문맥만 바뀌었고, 그 결과 공간의 기하가
바뀌었다. 과제가 말하는 학습 목표 그대로다.

## 메모

- 코퍼스가 작아서 `random` seed를 42로 고정했다. 같은 seed면 위 숫자가 그대로
  재현된다.
- 학습 데이터가 38문장뿐이라 `most_similar` 결과에 `she`, `market` 같은
  기능어성 단어가 섞인다. 실제 word2vec은 수십억 단어로 학습한다는 점을
  감안할 것.
