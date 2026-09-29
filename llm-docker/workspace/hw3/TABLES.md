# HW#3 보고서용 표

`python make_tables.py` 로 생성. 숫자는 모델에서 직접 읽은 값이다.

## 표 0. 실험 설정

| 항목 | 값 |
| --- | --- |
| 모델 | Skip-gram + negative sampling (numpy 직접 구현) |
| 훈련 문장 | 38문장 (문제 4에서 8문장 추가) |
| 어휘 크기 | 61 |
| 임베딩 차원 | 32 |
| negative sample 수 | 5 (슬라이드 p.59 권장 5~20) |
| epochs / learning rate | 300 / 0.05 (선형 감쇠) |
| random seed | 42 |
| 초기화 | embedding·context 행렬 모두 랜덤 (슬라이드 p.45) |

## 표 1. 문제 1 - window 크기별 top-5 유사 단어

| 기준 단어 | 순위 | window=1 | 유사도 | window=4 | 유사도 |
| --- | --- | --- | --- | --- | --- |
| king | 1 | queen | 0.999 | prince | 0.907 |
|  | 2 | princess | 0.928 | queen | 0.832 |
|  | 3 | prince | 0.923 | princess | 0.740 |
|  | 4 | market* | 0.765 | kingdom | 0.512 |
|  | 5 | garden* | 0.761 | rules | 0.491 |
| queen | 1 | king | 0.999 | princess | 0.910 |
|  | 2 | princess | 0.934 | king | 0.832 |
|  | 3 | prince | 0.929 | prince | 0.728 |
|  | 4 | market* | 0.772 | kingdom | 0.471 |
|  | 5 | garden* | 0.769 | rules | 0.469 |
| dog | 1 | cat | 0.999 | cat | 0.753 |
|  | 2 | near | 0.782 | pet | 0.577 |
|  | 3 | morning* | 0.763 | runs | 0.572 |
|  | 4 | road* | 0.758 | feed | 0.563 |
|  | 5 | market* | 0.755 | garden | 0.563 |
| apple | 1 | she* | 0.851 | banana | 0.714 |
|  | 2 | banana | 0.710 | market | 0.644 |
|  | 3 | morning* | 0.657 | sweet | 0.624 |
|  | 4 | cat* | 0.648 | buys | 0.614 |
|  | 5 | dog* | 0.643 | eat | 0.596 |
| car | 1 | bus | 0.740 | wheels | 0.732 |
|  | 2 | work | 0.702 | take | 0.717 |
|  | 3 | sleeps* | 0.676 | parks | 0.611 |
|  | 4 | drive | 0.672 | road | 0.599 |
|  | 5 | garden* | 0.671 | drives | 0.589 |

`*` 는 해당 단어의 주제어 목록(코드의 `TOPICS`)에 없는 단어.

## 표 2. 문제 1 - top-5 안의 주제 적중 수

| 기준 단어 | window=1 | window=4 |
| --- | --- | --- |
| king | 3 / 5 | 5 / 5 |
| queen | 3 / 5 | 5 / 5 |
| dog | 2 / 5 | 5 / 5 |
| apple | 1 / 5 | 5 / 5 |
| car | 3 / 5 | 5 / 5 |
| **합계** | **12 / 25** | **25 / 25** |

## 표 3. 문제 2 - 선택한 쌍의 cosine similarity

| 쌍 | window=1 | window=4 | 비고 |
| --- | --- | --- | --- |
| king - queen | 0.999 | 0.832 | 같은 범주 |
| dog - cat | 0.999 | 0.753 | 같은 범주 |
| apple - banana | 0.710 | 0.714 | 같은 범주 |
| car - bus | 0.740 | 0.565 | 같은 범주 |
| king - banana | 0.397 | 0.212 | 다른 범주 |
| dog - car | 0.385 | 0.220 | 다른 범주 |

window=4 기준 최고: **king - queen (0.832)**, 최저: **king - banana (0.212)**

## 표 4. 문제 3 - PCA 평면에서의 그룹 분리도

|  | 그룹 내 평균거리 | 그룹 간 평균거리 | 비율 |
| --- | --- | --- | --- |
| window=1 | 0.232 | 2.718 | 11.71x |
| window=4 | 0.174 | 2.971 | 17.07x |

그림: `hw3_pca.png` (두 윈도우를 나란히 그린 것). PCA 축의 부호는 SVD 관례상 임의이므로 두 그림의 축 방향을 서로 비교하면 안 된다.

## 표 5. 문제 3 - 왕족 클러스터 내부의 남/여 구분 (window=4)

| 쌍 | cosine |
| --- | --- |
| king - prince | 0.907 |
| queen - princess | 0.910 |
| king - queen | 0.832 |
| king - princess | 0.740 |

2-D 투영에서는 한 축으로 정렬되지 않지만 cosine 값에는 남아 있다.

## 표 6. 문제 4 - 추가한 훈련 문장

| # | 문장 |
| --- | --- |
| 1 | apple company makes computer |
| 2 | apple makes smartphone |
| 3 | apple company makes phone |
| 4 | the apple company sells computer and phone |
| 5 | apple is a technology company |
| 6 | many people buy the apple smartphone |
| 7 | the company makes a new computer |
| 8 | i bought a new apple phone |

## 표 7. 문제 4 - `most_similar("apple")` 전후 비교

| 순위 | before (38문장) | 유사도 | after (46문장) | 유사도 |
| --- | --- | --- | --- | --- |
| 1 | banana | 0.714 | she | 0.489 |
| 2 | market | 0.644 | market | 0.487 |
| 3 | sweet | 0.624 | buys | 0.482 |
| 4 | buys | 0.614 | phone | 0.477 |
| 5 | eat | 0.596 | eat | 0.472 |

## 표 8. 문제 4 - `apple` 과의 유사도 변화

| 상대 단어 | 구분 | before | after |
| --- | --- | --- | --- |
| banana | 과일 문맥 | 0.714 | 0.434 |
| phone | 새 문맥 | 어휘에 없음 | 0.477 |
| computer | 새 문맥 | 어휘에 없음 | 0.469 |
| company | 새 문맥 | 어휘에 없음 | 0.426 |
| smartphone | 새 문맥 | 어휘에 없음 | 0.359 |
