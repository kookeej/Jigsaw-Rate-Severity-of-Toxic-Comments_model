🏆 Jigsaw Rate Severity of Toxic Comments
===
Kaggle☁ Jigsaw Rate Severity of Toxic CommentsRank relative ratings of toxicity between comments!      
악성 댓글의 독성 정도를 측정하여 score를 주는 프로젝트입니다.


# 1. Preprocessing
* `Ruddit dataset`을 사용하여 새로운 데이터셋을 만든다. 
* Ruddit dataset에는 severity 정도가 수치로 나타내져 있는데, 이 수치를 토대로 less toxic comments, more toxic comments로 나눈다.
* 예시는 아래와 같다.    
![image](https://user-images.githubusercontent.com/74829786/177865875-e9d7ad31-5171-40e0-a5ca-d20a9f901d67.png)


# 2. Model
* 두 개의 roberta-base 모델을 사용하여 less toxic comments와 more toxic comments의 score를 예측한다.
* 이 두 모델은 가중치를 공유한다.

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
