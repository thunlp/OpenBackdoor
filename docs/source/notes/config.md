# Config

OpenBackdoor suggests to use a `.json` configuration file to specify modules and hyperparameters. We provide several example configs in `configs` folder.

To use a config file, just run the code
```bash
python demo_attack.py --configs/base_config.json
```

The `base_config.json` looks like
```json
{
    "target_dataset":{
        "name": "sst-2",
        "dev_rate": 0.1
    },
    "poison_dataset":{
        "name": "sst-2",
        "dev_rate": 0.1
    },
    "victim":{
        "type": "plm",
        "model": "bert",
        "path": "bert-base-uncased",
        "num_classes": 2,
        "device": "gpu",
        "max_len": 512
    },
    "attacker":{
        "name": "base",
        "metrics": ["accuracy"],
        "train":{
            "name": "base",
            "lr": 2e-5,
            "weight_decay": 0,
            "epochs": 2,
            "batch_size": 32,
            "warm_up_epochs": 3,
            "ckpt": "best",
            "save_path": "./models"
        },
        "poisoner":{
            "name": "badnet",
            "poison_rate": 0.1,
            "target_label": 1
        }
    },

    "defender":{
        "name": "rap",
        "pre": false,
        "correction": false,
        "metrics": ["FRR", "FAR"]
    },

    "train":{
        "name": "base",
        "lr": 2e-5,
        "weight_decay": 0,
        "seed": 123,
        "epochs": 2,
        "batch_size": 4,
        "warm_up_epochs": 3,
        "ckpt": "best",
        "save_path": "./models"
    }
}
```
