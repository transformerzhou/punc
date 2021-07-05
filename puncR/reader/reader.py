from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from tqdm import tqdm


from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
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
        max_tokens: int = None,
        text_num: int = 50000,
        **kwargs
    ):
        super().__init__(manual_distributed_sharding = True ,
                         manual_multiprocess_sharding = True ,
                         **kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.text_num = text_num
        self.punc_dic = {'，','。','；','？','！'}

    def text_to_instance(self, text: str) -> Instance:
        tokens, labels = self.text_process(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens)
        fields = {"text": text_field}
        if labels:
            fields["label"] = SequenceLabelField(labels, text_field)
        return Instance(fields)
    
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["text"].token_indexers = self.token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, encoding='utf8') as lines:
            for line in self.shard_iterable(lines.readlines()[:self.text_num]):
                yield self.text_to_instance(line.strip())
                
                
    def text_process(self, text):
        tokens = self.tokenizer.tokenize(text)
        new_tokens = [tokens[0]]
        labels = [0]
        for i, token in enumerate(tokens[1:]):
            if token.text not in self.punc_dic:
                new_tokens.append(token)
                labels.append(0)
            else:
                labels[-1] = 1

        return new_tokens, labels

