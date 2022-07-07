import argparse
import pickle
from tqdm import tqdm
import gc
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import logging

from config import DefaultConfig
from model import CustomModel
from preprocessing import CustomDataset
from utils import get_criterion, get_optimizer, get_scheduler, seed_everything

from colorama import Fore, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN
r_ = Fore.RED
sr_ = Style.RESET_ALL

# Settings
cfg = DefaultConfig()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_everything(cfg.SEED)
logging.set_verbosity_error()



def train(train_dataloader, valid_dataloader, args):
    
    # 모델 로딩
    model = CustomModel(config=cfg.MODEL_CONFIG)
    model.parameters
    model.to(device)

    criterion = get_criterion()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, train_dataloader, args)


    gc.collect()
    train_total_loss = []
    valid_total_loss = []

    best_val_loss = np.inf
    best_val_acc = -1

    for epoch in range(args.epochs):
        model.train()
        print(f"{y_}[EPOCH {epoch+1}]{sr_}")

        # 학습 단계 loss/accuracy
        train_loss_value = 0
        train_epoch_loss = []

        # 검증 단계 loss/accuracy
        valid_loss_value = 0
        valid_epoch_loss = []


        gc.collect()
        train_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, items in enumerate(train_bar):
            input_ids = items['input_ids'].to(device, dtype=torch.long)
            attention_mask = items['attention_mask'].to(device, dtype=torch.long)

            input_ids2 = items['input_ids2'].to(device, dtype=torch.long)
            attention_mask2 = items['attention_mask2'].to(device, dtype=torch.long)

            targets = items['target'].to(device, dtype=torch.long)

            optimizer.zero_grad()

            less_outputs = model(input_ids, attention_mask)
            more_outputs = model(input_ids2, attention_mask2)
            loss = criterion(more_outputs, less_outputs, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_value += loss.item()
            if (idx + 1) % cfg.TRAIN_LOG_INTERVAL == 0:
                train_bar.set_description("Loss: {:3f}".\
                    format(train_loss_value/cfg.TRAIN_LOG_INTERVAL))
                train_epoch_loss.append(train_loss_value/cfg.TRAIN_LOG_INTERVAL)
                train_loss_value = 0

                train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))

        with torch.no_grad():
            print(f"{b_}---- Validation.... ----{sr_}")
            model.eval()
            valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
            for idx, items in enumerate(valid_bar):
                input_ids = items['input_ids'].to(device, dtype=torch.long)
                attention_mask = items['attention_mask'].to(device, dtype=torch.long)

                input_ids2 = items['input_ids2'].to(device, dtype=torch.long)
                attention_mask2 = items['attention_mask2'].to(device, dtype=torch.long)

                targets = items['target'].to(device, dtype=torch.long)


                less_outputs = model(input_ids, attention_mask)
                more_outputs = model(input_ids2, attention_mask2)
                loss = criterion(more_outputs, less_outputs, targets)

                valid_loss_value += loss.item()
                if (idx + 1) % cfg.VALID_LOG_INTERVAL == 0:
                    valid_bar.set_description("Loss: {:3f}".\
                        format(valid_loss_value/cfg.VALID_LOG_INTERVAL))
                    valid_epoch_loss.append(valid_loss_value/cfg.VALID_LOG_INTERVAL)
                    valid_loss_value = 0
                    valid_batch_acc = 0

            print("{}Best Loss: {:3f}    |    This epoch Loss: {:3f}".format(g_, best_val_loss, (sum(valid_epoch_loss)/len(valid_epoch_loss))))
            if best_val_loss > (sum(valid_epoch_loss)/len(valid_epoch_loss)):
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), cfg.MODEL_PATH)
                print(f"{r_}Best Loss Model was Saved!{sr_}")
                best_val_loss = (sum(valid_epoch_loss)/len(valid_epoch_loss))

            valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))
        print()
    del train_total_loss, valid_total_loss, train_loss_value, train_epoch_loss, valid_loss_value, valid_epoch_loss, train_bar, valid_bar


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    train_dataloader = pickle.load(open('data/train_dataloader.pkl', 'rb'))
    valid_dataloader = pickle.load(open('data/valid_dataloader.pkl', 'rb'))
    
    train(train_dataloader, valid_dataloader, args)