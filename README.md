<div align="center">

# U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster (ICML 2026)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2604.09041-b31b1b.svg)](https://arxiv.org/abs/2604.09041)

![U-Cast Forecast Visualization](https://github.com/salvaRC/assets/blob/main/u-cast/u-cast-example-forecast-q1000.gif)
Official code for our ICML 2026 paper with pretrained checkpoints and the full **two-stage training** pipeline.
</div>

## 🛠️ Environment

Install the package and all of its dependencies with pip:

```bash
pip install -e .
```

This installs everything declared in [`pyproject.toml`](pyproject.toml) (PyTorch, PyTorch
Lightning, xarray + zarr, Hydra, einops, TensorDict, wandb, gcsfs, ...), covering both inference
and training.

## 🌪️ Quickstart (inference)

Run out-of-the-box inference using a pretrained U-Cast checkpoint (downloaded from Hugging Face) applied to ERA5 data (downloaded from Google Cloud below)
 using 5 ensemble members on two initial condition start dates, computing the RMSE and CRPS scores, and uploading them to Weights & Biases:
```bash
python run_inference_standalone.py \
    --ckpt-path hf:salv47/u-cast/ucast.ckpt \
    --data-dir gs://weatherbench2/datasets/era5 \
    --ic-start-dates 2020-01-01 2020-07-04 \
    --ensemble-size 5 \
    --score \
    --wandb-project SOME_PROJECT_NAME_TO_UPLOAD_SCORES_TO
```

## 🚀 Inference

The main entry point is [`run_inference_standalone.py`](run_inference_standalone.py); see the docstring at the top of the file for full usage instructions. Pretrained U-Cast checkpoints are hosted on [Hugging Face](https://huggingface.co/salv47/u-cast/tree/main) and are downloaded automatically the first time the script is run.

## 🧠 Training

The training entry point is [`run.py`](run.py), driven by Hydra configs in [`configs/`](configs).
U-Cast follows the paper's **two-stage curriculum**: a long, cheap deterministic pre-training
stage on weighted MAE, then a short probabilistic fine-tuning stage on the CRPS.

Both stages read the WeatherBench-2 1.5° ERA5 zarr referenced by `datamodule.dataset`
(default: `1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`). The
`datamodule.data_dir` you pass must contain that zarr.

In the configs, fields set to `???` are **mandatory**: Hydra raises an error unless you provide
them. Supply them on the command line as `key=value` (replace the placeholder paths below with
your own).

### Stage 1: deterministic pre-training (weighted MAE)

Uses [`configs/config_det.yaml`](configs/config_det.yaml): a single forward pass per step, `loss_function=wmae`, 100 epochs...

```bash
# datamodule.data_dir: directory containing the ERA5 .zarr
python run.py --config-name config_det \
    datamodule.data_dir=/path/to/era5
```

This writes checkpoints to `results/checkpoints/<wandb-run-id>/` (or `${ckpt_dir}` without W&B).
Note the run-id / checkpoint path, since Stage 2 warm-starts from it.

### Stage 2: probabilistic CRPS fine-tuning

Uses [`configs/config_prob.yaml`](configs/config_prob.yaml), which **inherits everything from
`config_det.yaml`** and only overrides the probabilistic differences: `loss_function=wcrps`,
a 2-member training ensemble (`num_training_ensemble_members=2`), 8 epochs, and the path to the Stage-1 checkpoint to warm-start from.

```bash
# from_pretrained_local_path:        dir holding the Stage-1 checkpoint(s), or a direct .ckpt file
# from_pretrained_checkpoint_run_id: the Stage-1 run id (locates <local_path>/<run_id>/<filename>)
# from_pretrained_checkpoint_filename: filename within that dir (default last.ckpt; ignored if path is a file)
python run.py --config-name config_prob \
    datamodule.data_dir=/path/to/era5 \
    module.from_pretrained_local_path=/path/to/stage1/checkpoints \
    module.from_pretrained_checkpoint_run_id=stage1_run_id \
    module.from_pretrained_checkpoint_filename=last.ckpt
```

MC-Dropout (rate `model.dropout=0.1`) is the only source of stochasticity. It is active during
both stages and turns into the ensemble generator under the CRPS objective in Stage 2.

### Stage 3 (optional): deep ensemble

Repeat Stage 2 `K` times (the paper uses `K=4`) with different seeds, each warm-starting from the
**same** Stage-1 checkpoint, then combine the resulting checkpoints at inference:

```bash
for SEED in 11 22 33 44; do
  python run.py --config-name config_prob seed=$SEED \
      datamodule.data_dir=/path/to/era5 \
      module.from_pretrained_local_path=/path/to/stage1/checkpoints \
      module.from_pretrained_checkpoint_run_id=stage1_run_id
done
```

### Batch size

Both stages target an **effective (global) batch size of 48** (`datamodule.batch_size`). The
per-GPU micro-batch is `datamodule.batch_size_per_gpu`. If you hit OOM, lower
`batch_size_per_gpu` (e.g. `datamodule.batch_size_per_gpu=1`); the effective batch size stays 48.

### Common overrides

```bash
python run.py --config-name config_det trainer.devices=[0,1,2,3]   # multi-GPU
python run.py --config-name config_det datamodule.batch_size_per_gpu=2   # less GPU memory
python run.py --config-name config_prob ckpt_path=/path/to/ckpt.ckpt   # resume an interrupted run
python run.py --config-name config_prob eval_mode=validate ckpt_path=/path/to/ckpt.ckpt  # validation only
```

### Weights & Biases

By default the run is logged to W&B project `u-cast`. Set
`logger.wandb.entity=<your-entity>` (or `logger.wandb.mode=disabled`) to point logs at
your own account. The `WANDB_API_KEY` env var, if set, is used to log in automatically.


## 📚 Citation

```bibtex
@article{cachay2026ucast,
  title = {U-Cast: A Surprisingly Simple and Efficient Frontier AI Probabilistic Weather Forecaster},
  author = {Cachay, Salva Rühling and Watson-Parris, Duncan and Yu, Rose},
  journal = {International Conference on Machine Learning},
  year = {2026},
}
```
