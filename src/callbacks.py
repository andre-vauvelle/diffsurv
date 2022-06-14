import torchvision
from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt

from modules.sorter import CustomDiffSortNet
from omni.visualize import torch_show
import wandb
from modules.tasks import RiskMixin, SortingRiskMixin, _get_soft_perm


class LogPredictionsCallback(Callback):

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        if not batch_idx % 40:
            if isinstance(list(pl_module.modules())[0], SortingRiskMixin):
                wandb_logger = trainer.logger
                # `outputs` comes from `LightningModule.validation_step`
                loss, predictions, perm_prediction, perm_ground_truth = outputs
                # Let's log 20 sample image predictions from first batch
                label_times = batch['label_times']
                perm_ground_truth_asc, perm_prediction_asc = self.plot_permutations(label_times,
                                                                                    perm_ground_truth,
                                                                                    perm_prediction
                                                                                    )
                true_label_times = batch['future_label_times_uncensored']
                perm_ground_truth_asc_t, perm_prediction_asc_t = self.plot_permutations(true_label_times,
                                                                                        perm_ground_truth,
                                                                                        perm_prediction
                                                                                        )

                captions = ["Soft Permutation", "Predicted Permutation"]

                wandb_logger.log_image(key=f'batch:{batch_idx}', images=[perm_ground_truth_asc_t,
                                                                         perm_prediction_asc_t],
                                       caption=captions)

            else:
                wandb_logger = trainer.logger
                loss, predictions, logits, label_times, label_multihot = outputs
                lh = logits.squeeze()
                lh = predictions.squeeze()
                lh = lh * -1
                e = label_multihot.squeeze()
                d = label_times.squeeze()
                sorter = CustomDiffSortNet(sorting_network_type='odd_even', size=predictions.shape[0], steepness=50,
                                           distribution='cauchy')
                sort_out, perm_prediction = sorter(lh.unsqueeze(0))
                perm_ground_truth = _get_soft_perm(e, d)
                idx = torch.argsort(label_times.squeeze(), descending=False)
                perm_ascending = torch.nn.functional.one_hot(idx).transpose(-2, -1).float()
                perm_prediction_asc = perm_ascending.T @ perm_prediction
                perm_ground_truth_asc = perm_ascending.T @ perm_ground_truth
                torch_show([perm_ground_truth_asc, perm_prediction_asc])
                plt.show()

    @staticmethod
    def plot_permutations(times, perm_ground_truth, perm_prediction):
        """Reoders permutation form largest to the smallest times"""
        idx = torch.argsort(times.squeeze(), descending=False)
        perm_ascending = torch.nn.functional.one_hot(idx).transpose(-2, -1).float()
        perm_prediction_asc = perm_ascending.T @ perm_prediction
        perm_ground_truth_asc = perm_ascending.T @ perm_ground_truth
        torch_show([perm_ground_truth_asc, perm_prediction_asc])
        plt.show()
        return perm_ground_truth_asc, perm_prediction_asc


def debug_plot():
    pass
    # lt_dist = torch.nn.functional.normalize(label_times.squeeze()[idx], dim=0)
    # pred_dist = torch.nn.functional.normalize(predictions[idx], dim=0)

    # data = list(zip(lt_dist, pred_dist))
    # table = wandb.Table(data=data, columns=["label", "value"])
    # wandb_logger.({"my_bar_chart_id": wandb.plot.bar(table, "label", "value", title="Custom Bar Chart")})


log_predictions_callback = LogPredictionsCallback()
