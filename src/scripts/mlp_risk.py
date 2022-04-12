import torch
from pytorch_lightning.utilities.cli import LightningCLI

from data.datamodules import DataModuleAssessmentRiskPredict
from modules.mlp import MultilayerRisk


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Automatically set model input dim based on data
        # TODO: remove symbols from label space?
        parser.link_arguments("data.input_dim", "model.input_dim", apply_on="instantiate")
        parser.link_arguments("data.output_dim", "model.output_dim", apply_on="instantiate")
        parser.link_arguments("data.label_vocab", "model.label_vocab", apply_on="instantiate")
        parser.link_arguments("data.grouping_labels", "model.grouping_labels", apply_on="instantiate")


# class CustomSaveConfigCallback(SaveConfigCallback):
#     def on_train_start(self, trainer, pl_module) -> None:
#         pass
#         # log_dir = trainer.log_dir or trainer.default_root_dir
#         # config_path = os.path.join(log_dir, self.config_filename)
#         # self.parser.save(self.config, config_path, skip_none=False)


def cli_main():
    cli = CustomLightningCLI(
        MultilayerRisk, DataModuleAssessmentRiskPredict, seed_everything_default=42,
        trainer_defaults={"gpus": -1 if torch.cuda.is_available() else 0},
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
