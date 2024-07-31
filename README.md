# TuneOut📻

<img src="logo.png" alt="TuneOut logo" width="200"/>

Commercial machine learning models are often trained on private or sensitive data. Understanding whether a machine learning model’s training data can be inferred from the model itself is of great interest for evaluating the risk of model leakage before public release, detecting copyright infringements in training data, and identifying LLM benchmark doping. While successful attacks have been developed for tabular and image models, text model attacks are still in their naissance. This study aims to develop membership inference attacks to determine whether specific text snippets were part of an LLM's training data. Our work expands on previous methods and achieves competitive performance using a new algorithm called TuneOut. TuneOut provides a robust classification by filtering erroneous outliers which impact the classification performance of many current methods. Problems with validation data and reproducibility are also identified and established methods are shown to perform considerably worse than previously believed on carefully curated datasets.

## Reproducing
```bash
$ pip install -r requirements.txt
$ python save_logits.py
    # usage: save_logits.py [-h] [-m MODEL_NAME] [-d DATASET_NAME] [-s SPLIT] [--label LABEL_KEY] [--sample SAMPLE_KEY]

    # options:
    #   -h, --help            show this help message and exit
    #   -m MODEL_NAME, --model MODEL_NAME
    #   -d DATASET_NAME, --dataset DATASET_NAME
    #   -s SPLIT, --split SPLIT
    #   --label LABEL_KEY
    #   --sample SAMPLE_KEY
```