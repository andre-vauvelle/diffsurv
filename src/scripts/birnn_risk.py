import torch

from data.datamodules import DataModuleAssessmentRiskPredict
from modules.bidirectional_rnn import BiRNNRisk
from scripts.cli_tools import RiskLightningCLI


def cli_main():
    cli = RiskLightningCLI(
        BiRNNRisk,
        DataModuleAssessmentRiskPredict,
        seed_everything_default=42,
        trainer_defaults={"gpus": -1 if torch.cuda.is_available() else 0},
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
