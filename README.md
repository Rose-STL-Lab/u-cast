<div align="center">

# U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2604.09041-b31b1b.svg)](https://arxiv.org/abs/2604.09041)

![U-Cast Forecast Visualization](https://github.com/salvaRC/assets/blob/main/u-cast/u-cast-example-forecast-q1000.gif)
</div>

## 🛠️ Environment

Inside a virtual environment, you can install the required packages using pip:

```bash
pip install torch xarray zarr einops tqdm pyyaml huggingface_hub wandb gcsfs
```

## 🌪️ Quickstart

Run out-of-the-box inference using a pretrained U-Cast checkpoint (downloaded from Hugging Face) applied to ERA5 data (downloaded from Google Cloud below)
 using 5 ensemble members on two initial condition start dates, computing the RMSE and CRPS scores, and uploading them to Weights & Biases:
```bash
python run_inference_standalone.py \
    --ckpt-path hf:salvaRC/u-cast/ucast.ckpt \
    --data-dir gs://weatherbench2/datasets/era5 \
    --ic-start-dates 2020-01-01 2020-07-04 \
    --ensemble-size 5 \
    --score \
    --wandb-project SOME_PROJECT_NAME_TO_UPLOAD_SCORES_TO
```

## 🚀 Inference

The main entry point is [`run_inference_standalone.py`](run_inference_standalone.py); see the docstring at the top of the file for full usage instructions. Pretrained U-Cast checkpoints are hosted on [Hugging Face](https://huggingface.co/salv47/u-cast/tree/main) and are downloaded automatically the first time the script is run.


## 🧠 Training

Please stay tuned for the training code, which will be released soon.


## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{cachay2026ucast,
  title = {U-Cast: A Surprisingly Simple and Efficient Frontier AI Probabilistic Weather Forecaster},
  author = {Cachay, Salva Rühling and Watson-Parris, Duncan and Yu, Rose},
  journal = {International Conference on Machine Learning},
  year = {2026},
}
```