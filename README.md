# Versatile Cardiovascular Signal Generation with a Unified Diffusion Transformer

[![DOI](https://zenodo.org/badge/989983076.svg)](https://doi.org/10.5281/zenodo.17240784)
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/thu-ml/UniCardio/blob/master/LICENSE)

Cardiovascular signals such as photoplethysmography (PPG), electrocardiography (ECG), and blood pressure (BP) are inherently correlated and complementary, together reflecting the health of cardiovascular system. However, their joint utilization in real-time monitoring is severely limited by diverse acquisition challenges from noisy wearable recordings to burdened invasive procedures. Here we propose UniCardio, a multi-modal diffusion transformer that reconstructs low-quality signals and synthesizes unrecorded signals in a unified generative framework. Its key innovations include a specialized model architecture to manage the signal modalities involved in generation tasks and a continual learning paradigm to incorporate varying modality combinations. By exploiting the complementary nature of cardiovascular signals, UniCardio clearly outperforms recent task-specific baselines in signal denoising, imputation, and translation. The generated signals match the performance of ground-truth signals in detecting abnormal health conditions and estimating vital signs, even in unseen domains, while ensuring interpretability for human experts. These advantages position UniCardio as a promising avenue for advancing AI-assisted healthcare.

### Requirements

- pytorch-cuda=12.4
- cuda-version=12.4

To install the necessary packages:

```
pip install -r requirements.txt
```
or

```
conda env create -f environment.yml
```

### Download Dataset

Cuff-Less Blood Pressure Estimation
https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation

### Execution

```
python train_original.py
```
This is the code to train UniCardio.

### Inference

The dataloader generate four versions of one batch

The first three batches are all of shape shape (B, 1, 2000), where B is the batch size

The first batch is four modalitites concatenated together, each of length 500. They are PPG, BP, ECG and zeros for the auxiliary modality.

The second batch consists of the noisy version of the clean first batch signal.

The third batch contains the intermittent version of the clean first batch.

The fourth batch contains the masks for this intermittent signals, indicating where to be imputed.

The flag determines what tasks to be performed. 


In the test_final.py, there are examples on testing on common tasks using DDPM or DDIM sampler.
