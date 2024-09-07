```markdown
# Dacon FSI Competition

[Dacon Competition Link](https://dacon.io/competitions/official/236297/overview/description)

## Docker Image

Docker Image From: [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch/tags)

Use tag with `-devel`, torch `>=2.0`.

Docker is not required, but you should create your own environment.

## Commands

### For Training Classifier

```sh
cd classifier/
python train.py
```

### For Generating Data

```sh
cd generator/
python generate_ctgan.py
```

## ToDos

- Use baseline code (Build it with PyTorch)
- Currently, DP & DDP for multi-GPUs are not ready, To Do.
- EDA and find out which columns affect which fraud
- Develop SimpleNN into new models for Classifier

## Goals

- Build models and aggregate them
```


12 Fraud Types
https://complyadvantage.com/insights/types-of-financial-fraud/