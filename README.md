# MARE: Multimodal Analogical Reasoning for Disease Evolution-Aware Radiology Report Generation

## Introduction
Radiology report generation from longitudinal medical data is critical for assessing disease progression and automating diagnostic workflows. While recent methods incorporate longitudinal information, they primarily rely on multimodal feature fusion, with limited capacity for explicit disease evolution modeling and temporal reasoning. To address this, we propose MARE, an end-to-end framework that formulates longitudinal radiology report generation as a multimodal analogical reasoning task. Inspired by the Abduction–Mapping–Induction paradigm, MARE models latent relational structures underlying disease evolution by aligning lesion-level visual features across time and mapping them to the textual domain for temporally coherent and clinically meaningful report generation. To mitigate the spatial misalignment caused by patient positioning or imaging variation, we introduce an Adaptive Region Alignment (ARA) module for robust temporal correspondence. Additionally, we design Dual Evolution Consistency (DEC) losses to regularize analogical reasoning by enforcing temporal coherence in both visual and textual evolution paths. Extensive experiments on the Longitudinal-MIMIC dataset demonstrate that MARE significantly outperforms state-of-the-art baselines across both natural language generation and clinical effectiveness metrics, highlighting the value of structured analogical reasoning for disease evolution-aware report generation. 

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
cd MARE
pip install -r requirements.txt
```

**2. Prepare the training dataset**

Longitudinal-MIMIC: you can download this dataset from [here](https://github.com/CelestialShine/Longitudinal-Chest-X-Ray) and download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.

### Training

```bash
bash scripts/6-1.deep_run.sh
```

### Testing (For MIMIC-CXR)

```bash
bash scripts/6-2.deep_test.sh
```

## Acknowledgement

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) Some codes of this repo are based on R2GenGPT.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
