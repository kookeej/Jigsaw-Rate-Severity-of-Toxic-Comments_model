Jigsaw Rate Severity of Toxic Comments
===
### *[Kaggle] Jigsaw Rate Severity of Toxic Comments Rank relative ratings of toxicity between comments [[Link]](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/overview/evaluation)*    
* 악성 댓글의 독성(toxic) 정도를 측정하여 점수(score)를 내고 랭킹을 매기는 대회입니다.
* `Deep metric learning`을 사용하여 입력값으로 들어가는 댓글들 사이의 유사도를 측정하여 상대적인 독성(toxic) 정도를 나타냅니다.    

### ✔ Approach
* 별도의 metric이나 label이 주어지지 않고 less toxic, more toxic으로 나눈 데이터셋이 주어집니다.
* `Ruddit` 데이터셋과 이전 jigsaw의 다른 competition의 데이터셋도 사용할 수 있다고 명시되어있기 때문에 해당 데이터셋을 활용하여 (less toxic, more toxic) 쌍으로 만들었습니다.
* 어떤 기준으로 less/more toxic쌍을 생성하는지에 따라 모델의 성능이 크게 좌우되기 때문에 다양한 실험을 진행하였습니다.
* Metric learning을 통해 독성이 더 높은 댓글이 더 높은 수치를, 독성 정도가 더 낮은 댓글은 더 낮은 수치를 갖도록 학습시켰습니다. Ranking loss를 통해 입력 데이터(less toxic, more toxic comments) 사이의 상대적 거리를 학습하고 순위를 매겼습니다.

### ✔ Deep metric learning이란?
* `Metric`은 `distance`와 같은 의미입니다. 딥러닝 모델을 사용하여 거리 공간(distance space, metric space)를 학습시키고, metric space에 사상된 개체들은 유사한 개체끼리는 가까이, 유사하지 않은 개체끼리는 멀리 위치하도록 합니다.

### ✔ Margin Ranking Loss
* positive/negative pair를 이용하여 상대적인 위치를 학습시키고 입력 데이터를 랭킹시킵니다.
* 하지만 positive/negative pair에 민감하게 반응하기 때문에 많은 실험을 통해 가장 적절한 pair dataset을 만들어야 합니다.
* 이번 competition에서는 파이토치의 `Margin Ranking Loss`를 사용하였습니다.
    * $loss(x_1,x_2,y)=max(0,−y∗(x_1−x_2)+margin)$
    
---
## 1. Preprocessing
* Kaggle Competition Code에 올라온 [베이스라인 코드](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter)의 데이터 전처리 방식을 참고했습니다.
* `Ruddit dataset`을 사용하여 less toxic, more toxic pair를 만들었습니다. 이때 어떻게 pair를 구성했는지에 따라 모델의 성능이 크게 좌우되기 때문에 많은 실험을 통해 최적의 pair를 찾아야합니다.
* Ruddit dataset에는 severity 정도가 수치로 나타내져 있는데, 이 수치를 토대로 less toxic comments, more toxic comments로 나누어줍니다.
* 데이터셋 예시는 아래와 같습니다.    
![image](https://user-images.githubusercontent.com/74829786/177865875-e9d7ad31-5171-40e0-a5ca-d20a9f901d67.png)

---

## 2. Model
* RoBERTa-base의 bi-encoder구조를 가진 모델입니다. 이 두 개의 인코더는 서로 가중치를 공유합니다.
* less toxic comments와 more toxic comments를 각각의 인코더의 입력값으로 넣은 뒤, 두 출력값의 [CLS] 토큰을 fully-connected layer를 통과시킨 후 metric learning을 진행합니다. 

---

## 3. Run
### 3.1. Data Preprocessing
```python
$ python preprocessing.py \
  --train_path =TRAIN_DATASET_PATH
  --test_size  =0.1
  --max_len    =MAX_TOKEN_LENGTH  # 256
```

### 3.2. Training
```python
$ python train.py \
  --epochs =10
```
---

#### Reference
*[https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter)*
