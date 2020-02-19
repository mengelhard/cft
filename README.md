# Neural Conditional Event Time Models

This repository contains code needed to reproduce results from the paper of the same name.

# Dependencies

This project uses Python 3.6.9. Notable dependencies are:

- tensorflow 1.10
- numpy 1.14.5
- matplotlib
- scikit-learn

A complete list of packages may be found in `requirements.txt`, and may be installed as follows:

```
pip install -r requirements.txt
```

# Usage

Models and training scripts may be found in the `src` folder. If you have questions, please open an issue or email m.engelhard@duke.edu

## Sythetic Data

Results on the synthetic dataset we describe may be obtained by running `src/model.py`. Results for baselines models may be obtained by running `src/baselines.py`.

## Reddit

Reddit data must first be downloaded through the pushshift.io API. Results may then be obtained by running `src/train_reddit.py`; data formatting may be inferred from the `load_batch` function. Results for baseline models may be obtained by running `src/reddit_baselines.py`.

## MIMIC-III

MIMIC data must first be requested and downloaded from https://mimic.physionet.org. Results may then be obtained by running `src/train_mimic.py`; data formatting may be inferred from the `load_batch` function. Results for baseline models may be obtained by running `src/mimic_baselines.py`.
