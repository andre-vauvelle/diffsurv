# diffsurv

## Install
```{bash}
conda create -n diffsurv python=3.8.5
cd /path/to/diffsurv/
conda env update -f src/requirements.yaml
```

## Run experiments

Using pytorch-lightning it's easy to run the scripts.

```{bash}
cd /path/to/diffsurv/src/
conda activate diffsurv
python scripts/mlpdiffsort_synthetic.py --config jobs/configs/mlpdiffsort_synthetic.yaml
```

## Run Sweeps

```{bash}
wandb sweep jobs/configs/sweeps/mlpdiffsort_synthetic_sweep.yaml
conda activate diffsurv
wandb agent <sweep-id>
```

## Extract results
Ensure that the OnTrainEndResults callback is on. This will automatically save an wandb artifact with a parquet of results, logits, event times, risk and covariates.

If you've a model already trained you can run something like:
```{bash}
python scripts/mlpdiffsort_synthetic.py predict --config jobs/configs/mlpdiffsort.yaml --ckpt_path path/to/model.ckpt
```

To extract results for the predict_dataloader and a specified checkpoint. Also saved an artifact.

