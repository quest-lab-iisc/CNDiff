# ⏳✨ Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting 📈🌊

[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue.svg)](https://icml.cc/)

Welcome to the official implementation of our **ICML 2025** paper:  
> **Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting**  

🎯 Our method blends **Generative Model framework** 🌀 with **non-linear data transformations** 🔄 to unlock **state-of-the-art** forecasting performance across diverse time series datasets.  
Whether it’s climate 🌦, finance 💹, or energy ⚡ — this repo has you covered.

---

## 📜 Table of Contents
- [🚀 OpenReview](https://openreview.net/forum?id=kcUNMKqrCg)
- [📚 Paper](https://openreview.net/attachment?id=kcUNMKqrCg&name=pdf)
- [📂 Dataset](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
- 🛠 Installation
- 💻 Usage
- 🤝 Citation
- 📬 Contact

---

## 📚 Paper
📄 **ICML 2025** — *Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting*  
[📥 Read the Paper (openreview version)](https://openreview.net/attachment?id=kcUNMKqrCg&name=pdf)

---

## 📂 Datasets

Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) link and keep them in data folder

Note : datasets link is from [Autoformer](https://github.com/thuml/Autoformer) paper

---


## 🛠 Installation
Clone the repo and install dependencies 🐍:
```bash
git clone this repo
cd cndiff
pip3 install -r requirements.txt 
```
---

## 💻 Usage
### To run for all the datasets
```sh
chmod +x ./scripts/run_all.sh
./scripts/run_all.sh
```

### To run for each dataset
```sh
python3 -m scripts.run_cndiff --cfg ./< yaml file >
eg: python3 -m scripts.run_cndiff --cfg ./exchange.yaml
```

---

## 🤝 Citation

If you find this work useful, please cite our paper.
```
@inproceedings{rishiconditional,
  title={Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting},
  author={Rishi, J and Mothish, GVS and Subramani, Deepak},
  booktitle={Forty-second International Conference on Machine Learning}
}

```

## 📬 Contact

- Rishi J (rishij@iisc.ac.in)
- GVS Mothish (mothishg@iisc.ac.in)
- Deepak NS (deepakns@iisc.ac.in)
