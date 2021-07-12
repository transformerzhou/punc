local bert_model = 'hfl/chinese-roberta-wwm-ext-large';
#local bert_model = 'hfl/chinese-roberta-wwm-ext';

{
    "dataset_reader" : {
        "type": "data_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        },
        "max_tokens": 512,
        "text_num": 5000000
    },
    "train_data_path": "data/BCUT/train.txt",
    "validation_data_path": "data/BCUT/test.txt",
    "model": {
        "type": "tagger",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
    },
    "data_loader": {
        "batch_sampler":{
        "type": "bucket",
        "batch_size": 8,
        "sorting_keys": ['text']
        }, 
        "num_workers": 8
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 100,
        "patience": 20,
        "validation_metric":"+f1",
        "cuda_device": 1
    },
#     "distributed": {
#        "cuda_devices": [0, 1]
#    }

}