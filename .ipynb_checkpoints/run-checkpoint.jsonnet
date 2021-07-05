local bert_model = 'hfl/chinese-bert-wwm';

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
        "text_num": 500
    },
    "train_data_path": "data/train.txt",
    "validation_data_path": "data/dev.txt",
    "model": {
        "type": "tagger",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        }
    },
    "data_loader": {
        "batch_size": 16,
        "shuffle": true, 
        "num_workers": 8
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 20,
        "patience": 5,
    },
    "distributed": {
        "cuda_devices": [0, 1]
    }
}