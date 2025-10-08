<div align="center">
  <div>
    <h1>
        Versatile Cardiovascular Signal Generation with a Unified Diffusion Transformer
    </h1>
  </div>

  <div>
    Yuyang Miao</strong>,  
    Zehua Chen</strong></a>,   
    <a href="https://lywang3081.github.io/">Liyuan Wang</a>,
    Luyun Fan,
    Danilo Mandic,
    and Jun Zhu
  </div>

  <br/>
  <br/>
</div>

---

Cardiovascular signals such as photoplethysmography (PPG), electrocardiography (ECG), and blood pressure (BP) are inherently correlated and complementary, together reflecting the health of cardiovascular system. However, their joint utilization in real-time monitoring is severely limited by diverse acquisition challenges from noisy wearable recordings to burdened invasive procedures. Here we propose UniCardio, a multi-modal diffusion transformer that reconstructs low-quality signals and synthesizes unrecorded signals in a unified generative framework. Its key innovations include a specialized model architecture to manage the signal modalities involved in generation tasks and a continual learning paradigm to incorporate varying modality combinations. By exploiting the complementary nature of cardiovascular signals, UniCardio clearly outperforms recent task-specific baselines in signal denoising, imputation, and translation. The generated signals match the performance of ground-truth signals in detecting abnormal health conditions and estimating vital signs, even in unseen domains, while ensuring interpretability for human experts. These advantages position UniCardio as a promising avenue for advancing AI-assisted healthcare.

The official implementation codes are here.

![](framework.png)

---

## 🔧 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

### Cuffless BP 

Download from [Cuffless BP ](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation).


### PTBXL

Download from [PTBXL](https://physionet.org/content/ptb-xl/1.0.3/)

### MIMIC

Download from [MIMIC](https://physionet.org/content/mimicdb/1.0.0/)

### MIMIC PERform AF 

Download from [MIMIC PERform AF ](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/)

### WESAD

Download from [WESAD ](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)


---

## 🧠 Pretrained Model

Download [the pretrained model](https://www.dropbox.com/scl/fo/4tnumdlwg48fcurk1bxnp/AMrluHHcl3xuLrgoriJfAu8?rlkey=nn9z0t7l5j8254uze4o53xnte&st=qvetpwvu&dl=0)
and place it in:

```bash
UniCardio/base_model/no_compress799.pth
```

---

## 🚀 Training

UniCardio is using **dataparallel** training and can be adopted to **distributed**.

```bash
python train_original.py
```

---

## 🧩 Evaluation

To test a pretrained model:

```bash
python test_final.py
```
---

## 🧭 Key Highlights of UniCardio

- **Unified Generative Framework:**  
  A single model that performs versatile tasks like signal denoising, imputation, and translation across multiple cardiovascular signals (e.g., PPG, ECG, and BP).
- **Multi-modal Diffusion Transformer:**  
  Leverages a transformer-based diffusion model to capture complex relationships between different cardiovascular signals within a unified latent space for flexible generation.
- **Specialized Architecture:**  
  Employs modality-specific encoders and decoders to handle distinct signal types and uses task-specific attention masks to precisely control the information flow between modalities for each specific task.
- **Continual Learning Paradigm:**  
  Introduces a training approach that incorporates tasks with an increasing number of conditional signals in phases, effectively overcoming catastrophic forgetting and balancing complex multi-modal relationships.

---

---

## 📬 Contact

If you encounter issues or wish to discuss collaborations, please contact **Yuyang Miao**(ym520@ic.ac.uk) or **Liyuan Wang**.

