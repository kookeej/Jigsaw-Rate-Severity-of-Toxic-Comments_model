import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel

import gc

from config import DefaultConfig

cfg = DefaultConfig()



class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(cfg.MODEL_NAME, config=cfg.MODEL_CONFIG)
        self.sequential = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 1)
        )
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, 1)
        
    def forward(self, ids, mask):
        outputs = self.model(input_ids=ids, attention_mask=mask,
                        output_hidden_states=False)
        outputs = self.sequential(outputs[1])
        return outputs