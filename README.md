<h1 align='center'>Diffsurv: Differentiable sorting for censored time-to-event data. <br> (NeurIPS 2023)<br>
    [<a href="(https://proceedings.neurips.cc/paper_files/paper/2023/file/d1a25d7e93f06cb422b3a74a0aa3bf3f-Paper-Conference.pdf">paper</a>] </h1>

## Abstract

**TLDR**; Diffsurv is a novel method that extends differentiable sorting to handle partial orderings, a key challenge in real-world applications like survival analysis. It predicts possible permutations, accommodating uncertainty from censored samples.

Survival analysis is a crucial semi-supervised task in machine learning with significant real-world applications, especially in healthcare. The most common approach to survival analysis, Cox’s partial likelihood, can be interpreted as a ranking model optimized on a lower bound of the concordance index. We follow these connections further, with listwise ranking losses that allow for a relaxation of the pairwise independence assumption. Given the inherent transitivity of ranking, we explore differentiable sorting networks as a means to introduce a stronger transitive in8 ductive bias during optimization. Despite their potential, current differentiable sorting methods cannot account for censoring, a crucial aspect of many real-world datasets. We propose a novel method, Diffsurv, to overcome this limitation by extending differentiable sorting methods to handle censored tasks. Diffsurv predicts matrices of possible permutations that accommodate the label uncertainty introduced by censored samples. Our experiments reveal that Diffsurv outperforms established baselines in various simulated and real-world risk prediction scenarios. Furthermore, we demonstrate the algorithmic advantages of Diffsurv by presenting a novel method for top-k risk prediction that surpasses current methods. In conclusion, Diffsurv not only provides a novel framework for survival analysis through differentiable sorting, but also significantly impacts real-world applications by improving risk stratification and offering a methodological foundation for developing predictive models in healthcare and beyond.


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
python scripts/mlpdiffsort_synthetic.py predict --config jobs/configs/mlpdiffsort_synthetic.yaml --ckpt_path <path/to/checkpoints>/checkpoints/epoch=15-step=12800.ckpt
```

To extract results for the predict_dataloader and a specified checkpoint. Also saved an artifact.

-----

## Citation
```bibtex
@article{vauvelle2024differentiable,
  title={Differentiable sorting for censored time-to-event data.},
  author={Vauvelle, Andre and Wild, Benjamin and Eils, Roland and Denaxas, Spiros},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

