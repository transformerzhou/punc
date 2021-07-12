from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from tqdm import tqdm
import torch

from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, SequenceLabelField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@DatasetReader.register('data_reader')
class PuncRestoreReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = 512,
        text_num: int = None,
        **kwargs
    ):
        super().__init__(manual_distributed_sharding = True ,
                         manual_multiprocess_sharding = True ,
                         **kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.text_num = text_num

    def text_to_instance(self, text: str) -> Instance:
        tokens, labels, cut_list = self.text_process(text)

        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens)
        fields = {"text": text_field}
        cut_field = ArrayField(torch.tensor(cut_list, dtype=torch.long))
        fields['cut'] = cut_field
#         label_mask = LabelField(len(labels), skip_indexing=True)
#         fields['label_mask'] = label_mask
        if labels:
            fields["label"] = ArrayField(torch.tensor(labels, dtype=torch.long))
#             fields["label"] = TensorField(labels)
        return Instance(fields)
    
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["text"].token_indexers = self.token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, encoding='utf8') as lines:
            for line in self.shard_iterable(lines.readlines()[-self.text_num:]):
                yield self.text_to_instance(line.strip())
                
                
    def text_process(self, text):
        tokens = self.tokenizer.tokenize(text)
        word_list = text.split(' ')
        token_list = [self.tokenizer.tokenize(x)[1:-1] for x in word_list]
        
        new_tokens = [tokens[0]]
        labels = [0]
        cut_list = [0]
        
        for token in token_list:
            if not token:
                continue
            if token[0].text != '，':
                cut_list.append(cut_list[-1]+len(token)+1)
                new_tokens += token+[tokens[-1]]
                labels += [0]*(len(token)+1)
            else:
                labels[-1] = 1
                
        cut_list = cut_list[1:]
        
        cut_idx = [1 if i in cut_list else 0 for i in range(len(new_tokens))]
        
        assert len(cut_idx)==len(new_tokens)==len(labels)
        
        return new_tokens, labels, cut_idx
