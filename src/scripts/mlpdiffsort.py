import torch
from pytorch_lightning.cli import ArgsType, LightningCLI

from data.datamodules import DataModuleRisk
from modules.mlp import MultilayerDiffsort, MultilayerRisk


class DiffsortLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Automatically set model input dim based on data
        # TODO: remove symbols from label space?
        parser.link_arguments("data.input_dim", "model.input_dim", apply_on="instantiate")
        parser.link_arguments("data.cov_size", "model.cov_size", apply_on="instantiate")
        parser.link_arguments("data.output_dim", "model.output_dim", apply_on="instantiate")
        parser.link_arguments("data.label_vocab", "model.label_vocab", apply_on="instantiate")
        parser.link_arguments(
            "data.grouping_labels", "model.grouping_labels", apply_on="instantiate"
        )
        # parser.link_arguments("data.batch_size", "model.sorter_size", apply_on="instantiate")
        parser.link_arguments("data.risk_set_size", "model.sorter_size", apply_on="instantiate")
        parser.link_arguments("data.setting", "model.setting", apply_on="instantiate")

    # def before_fit(self):
    #     self.trainer.logger.experiment.watch(
    #         self.model,
    #         log="all",
    #     )


def diffsort_cli_main(args: ArgsType = None, run=True):
    torch.set_float32_matmul_precision("medium")

    cli = DiffsortLightningCLI(
        MultilayerDiffsort,
        DataModuleRisk,
        seed_everything_default=42,
        trainer_defaults={"gpus": -1 if torch.cuda.is_available() else 0},
        save_config_callback=None,
        args=args,
        run=run,
    )
    return cli


if __name__ == "__main__":
    cli = diffsort_cli_main()
    cli.trainer.test(ckpt_path="best", dataloaders=cli.datamodule.test_dataloader())
