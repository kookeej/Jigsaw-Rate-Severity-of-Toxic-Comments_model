Jigsaw Rate Severity of Toxic Comments
===
### *[Kaggle] Jigsaw Rate Severity of Toxic Comments Rank relative ratings of toxicity between comments [[Link]](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/overview/evaluation)*    
* 악성 댓글의 독성(toxic) 정도를 측정하여 점수(score)를 내고 랭킹을 매기는 대회입니다.
* `Deep metric learning`을 사용하여 입력값으로 들어가는 댓글들 사이의 유사도를 측정하여 상대적인 독성(toxic) 정도를 나타냅니다.    

### ✔ Deep metric learning이란?
* `Metric`은 `distance`와 같은 의미입니다.
* 딥러닝 모델을 사용하여 거리 공간(distance space)을 학습합니다. 학습된 metric space에서 유사한 개체끼리는 가까이, 유사하지 않은 개체끼리는 멀리 사상됩니다.


### ✔ 왜 Metric learning을 사용하였는가?
* 이 대회는 별도의 metric 없이, annotator들이 매긴 랭킹을 평균 낸 값이 기준이 됩니다.
* 따라서 metric learning을 통해 독성이 더 높은 댓글이 더 높은 랭킹을 갖도록 학습시키고, 이를 통해 입력 댓글들의 임베딩이 학습되도록 모델을 설계하는 것이 더 적절하기 때문에 metric learning을 사용하게 되었습니다. 

### Margin Ranking Loss
* 하지만 좋은 positive/negative pair를 찾아야 하기 때문에, 데이터셋에 민감하게 반응한다.
* 실제로 여러 데이터셋 조합을 고려하여 실험을 진행하였다.

# 1. Preprocessing
* Kaggle Competition Code에 올라온 [베이스라인 코드](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter)의 데이터 전처리 방식을 참고했습니다.
* `Ruddit dataset`을 사용하여 새로운 데이터셋을 만들어줍니다.
* Ruddit dataset에는 severity 정도가 수치로 나타내져 있는데, 이 수치를 토대로 less toxic comments, more toxic comments로 나누어줍니다.
* 데이터셋 예시는 아래와 같습니다.    
![image](https://user-images.githubusercontent.com/74829786/177865875-e9d7ad31-5171-40e0-a5ca-d20a9f901d67.png)


# 2. Model
* 두 개의 roberta-base 모델을 사용하여 less toxic comments와 more toxic comments의 score를 예측합니다.
* 이 두 모델은 가중치를 공유합니다.

***


### 💡 실행 방법

#### 1. Data Preprocessing
```python
$ python preprocessing.py \
  --train_path =TRAIN_DATASET_PATH
  --test_size  =0.1
  --max_len    =MAX_TOKEN_LENGTH  # 256
```

#### 2. Training
```python
$ python train.py \
  --epochs =10
```


#### Reference
* [https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter)
