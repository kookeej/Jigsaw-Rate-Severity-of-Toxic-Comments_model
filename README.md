๐ Jigsaw Rate Severity of Toxic Comments
===
### *Kaggleโ Jigsaw Rate Severity of Toxic CommentsRank relative ratings of toxicity between comments!๐*    
์์ฑ ๋๊ธ์ ๋์ฑ ์ ๋๋ฅผ ์ธก์ ํ์ฌ score๋ฅผ ์ฃผ๋ ํ๋ก์ ํธ์๋๋ค.    
์ฌ๊ธฐ์๋ `Metric learning`์ ํตํด ์๋ ฅ ๋๊ธ๋ค์ ๋์ฑ ๋ญํน์ ์ ๋ํ๋ผ ์ ์๋ ์๋ฒ ๋ฉ์ ํ์ต์์ผฐ์ต๋๋ค.
๋ํ ๋งํฌ: [https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/overview/evaluation](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/overview/evaluation)    

### ๐ก Metric learning์ด๋?
* `Metric learning`์ด๋, ์๋ ฅ ๋ฐ์ดํฐ ์ฌ์ด์ ๊ฑฐ๋ฆฌ๋ฅผ ํ์ตํ๋ ๊ฒ์ ์๋ฏธํฉ๋๋ค.
* ์ด ๋ ์ฌ์ด์ ๊ฑฐ๋ฆฌ/์ ์ฌ๋๋ฅผ ์๊ณ  ์๋ค๋ฉด, ์ด๋ฅผ ๋ง์ถ์ด๋๊ฐ๋ ๊ณผ์ ์ ํตํด ์๋ ฅ ๋ฐ์ดํฐ๋ฅผ ์ ์ค๋ชํ๋ ์๋ฒ ๋ฉ์ ํ์ตํ๋ ๊ฒ์๋๋ค.

### ๐ก ์ Metric learning์ ์ฌ์ฉํ์๋๊ฐ?
* ์ด ๋ํ๋ ๋ณ๋์ metric ์์ด, annotator๋ค์ด ๋งค๊ธด ๋ญํน์ ํ๊ท  ๋ธ ๊ฐ์ด ๊ธฐ์ค์ด ๋ฉ๋๋ค.
* ๋ฐ๋ผ์ metric learning์ ํตํด ๋์ฑ์ด ๋ ๋์ ๋๊ธ์ด ๋ ๋์ ๋ญํน์ ๊ฐ๋๋ก ํ์ต์ํค๊ณ , ์ด๋ฅผ ํตํด ์๋ ฅ ๋๊ธ๋ค์ ์๋ฒ ๋ฉ์ด ํ์ต๋๋๋ก ๋ชจ๋ธ์ ์ค๊ณํ๋ ๊ฒ์ด ๋ ์ ์ ํ๊ธฐ ๋๋ฌธ์ metric learning์ ์ฌ์ฉํ๊ฒ ๋์์ต๋๋ค.

# 1. Preprocessing
* Kaggle Competition Code์ ์ฌ๋ผ์จ [๋ฒ ์ด์ค๋ผ์ธ ์ฝ๋](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter)์ ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ ๋ฐฉ์์ ์ฐธ๊ณ ํ์ต๋๋ค.
* `Ruddit dataset`์ ์ฌ์ฉํ์ฌ ์๋ก์ด ๋ฐ์ดํฐ์์ ๋ง๋ค์ด์ค๋๋ค.
* Ruddit dataset์๋ severity ์ ๋๊ฐ ์์น๋ก ๋ํ๋ด์ ธ ์๋๋ฐ, ์ด ์์น๋ฅผ ํ ๋๋ก less toxic comments, more toxic comments๋ก ๋๋์ด์ค๋๋ค.
* ๋ฐ์ดํฐ์ ์์๋ ์๋์ ๊ฐ์ต๋๋ค.    
![image](https://user-images.githubusercontent.com/74829786/177865875-e9d7ad31-5171-40e0-a5ca-d20a9f901d67.png)


# 2. Model
* ๋ ๊ฐ์ roberta-base ๋ชจ๋ธ์ ์ฌ์ฉํ์ฌ less toxic comments์ more toxic comments์ score๋ฅผ ์์ธกํฉ๋๋ค.
* ์ด ๋ ๋ชจ๋ธ์ ๊ฐ์ค์น๋ฅผ ๊ณต์ ํฉ๋๋ค.

***


### ๐ก ์คํ ๋ฐฉ๋ฒ

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
