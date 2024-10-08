import torch
from packaging import version
import pytorch_lightning
from pytorch_lightning.cli import ArgsType

if version.parse(pytorch_lightning.__version__) >= version.parse("1.7.0"):
    from pytorch_lightning.cli import LightningCLI
else:
    from pytorch_lightning.utilities.cli import LightningCLI

from data.datamodules import DataModuleRisk
from modules.mlp import MultilayerRisk


class MLPLightningCLI(LightningCLI):
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
        parser.link_arguments("data.setting", "model.setting", apply_on="instantiate")

    # def before_fit(self):
    #     self.trainer.logger.experiment.watch(
    #         self.model,
    #         log="all",
    #     )


# class CustomSaveConfigCallback(SaveConfigCallback):
#     def on_train_start(self, trainer, pl_module) -> None:
#         pass
#         # log_dir = trainer.log_dir or trainer.default_root_dir
#         # config_path = os.path.join(log_dir, self.config_filename)
#         # self.parser.save(self.config, config_path, skip_none=False)


def mlp_cli_main(args: ArgsType = None, run=True):
    cli = MLPLightningCLI(
        MultilayerRisk,
        DataModuleRisk,
        seed_everything_default=42,
        trainer_defaults={"gpus": -1 if torch.cuda.is_available() else 0},
        save_config_callback=None,
        args=args,
        run=run,
    )
    return cli


if __name__ == "__main__":
    cli = mlp_cli_main()
    cli.trainer.test(ckpt_path="best", dataloaders=cli.datamodule.test_dataloader())
