# U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster

## Environment

We recommend using a virtual environment to manage dependencies. You can install the required packages using pip:

```bash
pip install torch xarray zarr einops tqdm pyyaml huggingface_hub wandb gcsfs
```  

## Inference

The main entry point is [`run_inference_standalone.py`](run_inference_standalone.py); see the docstring at the top of the file for full usage instructions. Pretrained U-Cast checkpoints are hosted on [Hugging Face](https://huggingface.co/salv47/u-cast/tree/main) and are downloaded automatically the first time the script is run.

Example usage:

```bash
python run_inference_standalone.py \
    --ckpt-path hf:salvaRC/u-cast/ucast.ckpt \
    --data-dir gs://weatherbench2/datasets/era5 \
    --ic-start-dates 2020-01-01 2020-07-04 \
    --ensemble-size 5 \
    --score \
    --wandb-project SOME_PROJECT_NAME_TO_UPLOAD_SCORES_TO
```
## Training

Please stay tuned for the training code, which will be released soon.


## Citation

If you use this code in your research, please cite:

```bibtex
@article{cachay2026ucast,
  title = {U-Cast: A Surprisingly Simple and Efficient Frontier AI Probabilistic Weather Forecaster},
  author = {Cachay, Salva Rühling and Watson-Parris, Duncan and Yu, Rose},
  journal = {International Conference on Machine Learning},
  year = {2026},
}
```