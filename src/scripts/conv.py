import torch
from pytorch_lightning.cli import ArgsType, LightningCLI

from data.datamodules import DataModuleRisk
from modules.convnets import ConvRisk


class ConvLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Automatically set model input dim based on data
        # TODO: remove symbols from label space?
        parser.link_arguments("data.label_vocab", "model.label_vocab", apply_on="instantiate")
        parser.link_arguments(
            "data.grouping_labels", "model.grouping_labels", apply_on="instantiate"
        )
        parser.link_arguments("data.setting", "model.setting", apply_on="instantiate")


def diffsort_cli_main(args: ArgsType = None, run=True):
    cli = ConvLightningCLI(
        ConvRisk,
        DataModuleRisk,
        seed_everything_default=42,
        trainer_defaults={"gpus": -1 if torch.cuda.is_available() else 0},
        save_config_callback=None,
        args=args,
        run=run,
    )
    return cli


if __name__ == "__main__":
    diffsort_cli_main()
