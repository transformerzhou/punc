import tempfile
from typing import Dict, Iterable, List, Tuple
from overrides import overrides
import torch
import numpy as np
import random

from allennlp.common.util import JsonDict
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import  Metric

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     print('random_seed: ', seed)
# # 设置随机数种子
# setup_seed(608)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]



class F1(Metric):
    def __init__(self):
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10

    def __call__(self, probs, label):
        
        self.correct_num += torch.sum((probs==1)&(label==1)).cpu().numpy()
        self.predict_num += torch.sum(probs==1).cpu().numpy()
        self.gold_num += torch.sum(label==1).cpu().numpy()
        

    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        precision = self.correct_num / self.predict_num
        recall = self.correct_num / self.gold_num
        f1_score = 2 * precision * recall / (precision + recall)
        if reset:
            self.reset()
        return {'pre':precision, 'rec':recall, 'f1':f1_score}
    
    @overrides
    def reset(self):
        print("correct_num：{}，gold_num：{}， predict_num：{}".format(self.correct_num, self.gold_num, self.predict_num))
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10

@Model.register("tagger")
class PuncRestoreLabeler(Model):
    def __init__(
            self, vocab, embedder:TextFieldEmbedder, threshold=0.5
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.th = threshold
        self.classifier = torch.nn.Linear(self.embedder.get_output_dim(), 1)
        self.f1 = F1()

    def forward(
            self,
            text: TextFieldTensors, label: torch.Tensor = None, cut: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        mask = text['tokens']['mask']
        # Shape: (batch_size, num_tokens, embedding_dim)
        # print(self.embedder)
        encoded_text = self.embedder(text)
        logits1 = self.classifier(encoded_text)
        
        logits2 = torch.squeeze(logits1, dim=-1)
        probs = torch.sigmoid(logits2)
        probs[torch.where(probs>self.th)] = 1
        probs[torch.where(probs<=self.th)] = 0
        probs = probs*cut
        
        bz = cut.shape[0]
        sl = cut.shape[1]
        
        for i in range(bz):
            for j in range(sl-1, -1, -1):
                if cut[i][j]==1:
                    cut[i][i] = 0
                    break
        output = {"probs": probs}
        output['token'] = text['tokens']['token_ids']
        output['cut'] = cut
        
        if label is not None:
            self.f1(probs[:], label[:])
            #一定是cut来做加权，mask不收敛
            output["loss"] = (torch.nn.functional.binary_cross_entropy_with_logits(logits2, label.float(), reduction='none')*cut).sum()/(cut.sum())

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.f1.get_metric(reset)