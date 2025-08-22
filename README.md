# DivideMotion: Dual-Stream Conditional Diffusion for Text-to-3D Hand–Object Interaction

> **One‑liner**: Diffusion-based Transformer model for generating 3D hand–object interaction motions from natural language descriptions.



---

## 🧭 Table of contents


* [Model architecture](#-model-architecture)
* [Quickstart](#-quickstart)
* [Installation](#-installation)
* [Demo](#-Results)
* [Data](#-data)
* [Training](#-training)
* [Evaluation](#-evaluation)
* [Results](#-results)
* [Project structure](#-project-structure)
* [Roadmap](#-roadmap)
* [Status](#-status)
* [Acknowledgments](#-Acknowledgments)




## 🧱 Model architecture

![Architecture](assets/pipelinee.png)


---

## 🚀 Quickstart
----
```bash
# Clone and enter the folder
git clone https://github.com/anonym3dv/HOI-generation.git && cd HOI-generation

```

## 🛠️ Installation



```bash
source scripts/install.sh
```

------


## 🎬 Results

> **Qualitative results — DivideMotion.**  
> The four clips illustrate the model’s output diversity for the same text prompt: variations in execution **style** , **tempo** , **trajectory**  and **finger articulation**.  


| [![Demo 1](assets/demo1.gif)](assets/demo1.mp4) | [![Demo 2](assets/demo2.gif)](assets/demo2.mp4) | [![Demo 3](assets/demo3.gif)](assets/demo3.mp4) | [![Demo 4](assets/demo4.gif)](assets/demo4.mp4) |
|---|---|---|---|


---

## 📂 Data

Follow the exact data protocols of [Text2HOI](https://github.com/JunukCha/Text2HOI/tree/main).

Folder structure and preprocessing identical to the official repo.

We do not redistribute datasets — obtain them from their official sources and respect their licenses.

This project expects the same preprocessed files Text2HOI.


---

## 🏋️ Training

```bash
source scripts/train/train_texthoi.sh
```
or Download the pre-trained **[DivideMotion](<MODEL_LINK>)** and place it in checkpoint folder
---

## 📊 Evaluation

For evaluation, make sure to generate data **20 times per prompt** and download our **[text–motion matching model](<MODEL_LINK>)**.


```bash
python Evaluation/evaluation_data_generation.py                  #for Evaluation data on test set


# Universal evaluation template — replace <> with your values
python eval.py \
--model_dir <PATH_TO_MODEL_DIR> \       # model dir is the text-motion match model downloaded
--log_file <PATH_TO_LOG_FILE> \
--device_id <GPU_ID>                    # After moving to the Text-motion-match folder 
```

---


## 🗺️ Roadmap

* [ ] Release the training code
* [ ] text-motion checkpoints



## 🚧 Status

This repository is **under active construction**. Interfaces, training scripts, and docs may change frequently.

If you run into errors or have questions, please **open an issue** .

We appreciate your feedback — it helps us prioritize fixes and improvements.

---
## 🙏 Acknowledgments

This project is **heavily inspired** by [Text2HOI](https://github.com/JunukCha/Text2HOI) (Cha et al., 2024).  
We follow their data protocols and some implementations.  


---
