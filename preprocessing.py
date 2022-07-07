import argparse
import pickle
from tqdm import tqdm
import gc

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import DefaultConfig

config = DefaultConfig()


def make_ruddit_less_more(path):
    df = pd.read_csv(path)
    # 데이터프레임 칼럼 정리
    df['score'] = df['offensiveness_score']
    df['text'] = df['txt']
    df = df.drop(['txt', 'offensiveness_score', 'post_id', 'comment_id', 'url'], axis=1)

    # 결측치 제거를 위한 결측치 존재 인덱스 저장
    drop_index = df[df['text'] == ('[deleted]' or '[removed]')].index
    # 결측치 행 drop
    df = df.drop(drop_index)
    # 인덱스 재정렬
    df = df.reset_index().drop(['index'], axis=1)

    ruddit_dataset = pd.DataFrame(columns=['less_toxic', 'less_score', 'more_toxic', 'more_score'])
    ruddit_dataset['less_toxic'] = df['text'][df['score'] < 0].reset_index(drop=True)[:len(df[df['score'] >= 0])]
    ruddit_dataset['less_score'] = df['score'][df['score'] < 0].reset_index(drop=True)[:len(df[df['score'] >= 0])]
    ruddit_dataset['more_toxic'] = df['text'][df['score'] >= 0].reset_index(drop=True)
    ruddit_dataset['more_score'] = df['score'][df['score'] >= 0].reset_index(drop=True)


    drop_df = df[df['score'] < 0].reset_index(drop=True)[len(df[df['score'] >= 0]):]
    drop_df = drop_df.reset_index(drop=True)
    
    # threshold
    p = -0.212
    sdrop_df = pd.DataFrame(columns=['less_toxic', 'less_score', 'more_toxic', 'more_score'])
    sdrop_df['less_toxic'] = drop_df['text'][drop_df['score'] <= p].reset_index(drop=True)
    sdrop_df['less_score'] = drop_df['score'][drop_df['score'] <= p].reset_index(drop=True)
    sdrop_df['more_toxic'] = drop_df['text'][drop_df['score'] > p][:len(drop_df[drop_df['score'] <= p])].reset_index(drop=True)
    sdrop_df['more_score'] = drop_df['score'][drop_df['score'] > p][:len(drop_df[drop_df['score'] <= p])].reset_index(drop=True)


    ruddit = pd.concat([ruddit_dataset, sdrop_df], axis=0).reset_index(drop=True)
    ruddit.to_csv("data/ruddit_less_more_with_score.csv", index=False)
    
    return ruddit



# Tokenizer
def tokenizing(dataset, args, mode):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    less_toxic = dataset['less_toxic'].tolist()
    more_toxic = dataset['more_toxic'].tolist()
    length = len(dataset)

    tokenized = tokenizer(
        less_toxic,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=args.max_len,
        return_token_type_ids=False
    )
    
    tokenized2 = tokenizer(
        more_toxic,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=args.max_len,
        return_token_type_ids=False
    )
    for key, value in tokenized2.items():
        tokenized[key+"2"] = value
        
    return tokenized, length


# Dataset 구성.
class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset, length, mode):
        self.tokenized_dataset = tokenized_dataset
        self.mode = mode
        self.length = length
        self.target = 1

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        if self.mode == "train":
            item['target'] = torch.tensor(self.target, dtype=torch.long)
        return item

    def __len__(self):
        return self.length
    
    
def pro_dataset(dataset, batch_size, args, mode="train"):
    tokenized, length = tokenizing(dataset, args, mode=mode)
    custom_dataset = CustomDataset(tokenized, length, mode=mode)
    if mode == "train":
        OPT = True
    else:
        OPT = False
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=batch_size,
        shuffle=OPT,
        drop_last=OPT
    )
    return dataloader





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/ruddit_with_text.csv', help="train dataset path")
    parser.add_argument('--test_size', type=float, default=0.1, help="train/test split size")
    parser.add_argument('--max_len', type=int, default=128, help="max token length for tokenizing")

    args = parser.parse_args()
        
    dataset = make_ruddit_less_more(args.train_path)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=args.test_size, random_state=42)
    
    print("train dataset size: {}    |    valid dataset size: {}".format(len(train_dataset), len(valid_dataset)))
    
    print("Preprocessing dataset...")
    train_dataloader = pro_dataset(train_dataset, config.TRAIN_BATCH, args)
    print("complete!")
    valid_dataloader = pro_dataset(valid_dataset, config.VALID_BATCH, args)
    print("complete!")

    
    # Save DataLoader with pickle file.
    print("Save DataLoader...")
    gc.collect()
    pickle.dump(train_dataloader, open('data/train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()
    pickle.dump(valid_dataloader, open('data/valid_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)   
    print("Data Preprocessing Complete!")     







