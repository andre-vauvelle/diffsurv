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
python scripts/mlpdiffsort_synthetic.py --config jobs/configs/risk/mlpdiffsort_synthetic.yaml
```

## Run Sweeps

```{bash}
wandb sweep jobs/configs/risk/sweeps/mlpdiffsort_synthetic_sweep.yaml
conda activate diffsurv
wandb agent <sweep-id>
```

### Wandb 
[wandb project](https://wandb.ai/cardiors/diffsurv)
