import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset

import wandb
from definitions import DATA_DIR
from omni.common import create_folder


class SVHNMultiDigit(VisionDataset):
    """`Preprocessed SVHN-Multi <>`_ Dataset.
    Note: The preprocessed SVHN dataset is based on the the `Format 1` official dataset.
    By cropping the numbers from the images, adding a margin of :math:`30\\%` , and resizing to :math:`64\times64` ,
    the dataset has been preprocessed.
    The data split is as follows:
        * ``train``: (30402 of 33402 original ``train``) + (200353 of 202353 original ``extra``)
        * ``val``: (3000 of 33402 original ``train``) + (2000 of 202353 original ``extra``)
        * ``test``: (all of 13068 original ``test``)
    Each ```train / val`` split has been performed using
    ``sklearn.model_selection import train_test_split(data_X_y_tuples, test_size=3000 / 2000, random_state=0)`` .
    This is the closest that we could come to the
    `work by Goodfellow et al. 2013 <https://arxiv.org/pdf/1312.6082.pdf>`_ .
    Args:
        root (string): Root directory of dataset where directory
            ``SVHNMultiDigit`` exists.
        split (string): One of {'train', 'val', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop`` .
            (default = random 54x54 crop + normalization)
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    split_list = {
        "train": [
            "https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_train.p",
            "svhn-multi-digit-3x64x64_train.p",
            "25df8732e1f16fef945c3d9a47c99c1a",
        ],
        "val": [
            "https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_val.p",
            "svhn-multi-digit-3x64x64_val.p",
            "fe5a3b450ce09481b68d7505d00715b3",
        ],
        "test": [
            "https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_test.p",
            "svhn-multi-digit-3x64x64_test.p",
            "332977317a21e9f1f5afe7ef47729c5c",
        ],
    }

    def __init__(
        self,
        root,
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomCrop([54, 54]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        target_transform=None,
        download=False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        data = torch.load(os.path.join(self.root, self.filename))

        self.data = data[0]
        # loading gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = data[1].type(torch.LongTensor)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img.numpy(), (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


# def investigate_time_function():
# plt.scatter(numbers, lambda_exp_BX, s=1, c='r', alpha=0.1)
# plt.scatter(log_numbers, lambda_exp_BX, s=1, c='r', alpha=0.1)
# plt.show()
# print(T.mean())
#
# idx = torch.argsort(log_numbers)
# T = T[idx]
# log_numbers = log_numbers[idx]
# numbers = numbers[idx]


# plt.scatter(log_numbers, T, alpha=0.1, s=1)
# df = pd.DataFrame.from_dict({"log_numbers": log_numbers, 'T':T})
# medians = df.groupby('log_numbers')['T'].median()
# plt.scatter(log_numbers, T.argsort()/max(T.argsort()), alpha=0.1, s=1)
# plt.scatter(medians.index, medians, s=1)
# plt.scatter(medians.index, medians.argsort()/max(medians.argsort()), s=1)

# plt.scatter(numbers, T.argsort(), alpha=0.1, s=1)

# Oracle performance
# if calc_oracle:
#     k = 5
#     perm = torch.randperm(T.argsort().size(0))
#     idx = perm[:k]
#     plt.scatter(log_numbers[idx], T.argsort()[idx], alpha=1, s=5)
#
#     T_oracle= -np.log(0.5)/(lambda_exp_BX)
#     plt.scatter(log_numbers, T_oracle, alpha=0.1, s=1)
#     plt.scatter(log_numbers, T_oracle.argsort()/max(T_oracle.argsort()), alpha=0.1, s=1)
#     plt.show()
#
#     samples = 1_000
#     k = 16
#     EM_total = 0
#     EW_total = 0
#     for _ in range(samples):
#         perm = torch.randperm(T.argsort().size(0))
#         idx = perm[:k]
#         rank_sample = T.argsort()[idx]
#         rank_oracle = T_oracle.argsort()[idx]
#         matches = rank_sample.argsort() == rank_oracle.argsort()
#         if matches.all():
#             EM_total += 1
#         EW_total += matches.sum()
#
#     EM = EM_total/samples
#     EW = EW_total/(k*samples)
#     print(f'EM{k}: {EM}, EW{k}: {EW}')

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torchvision.transforms as transforms

    beta = 500
    censored_proportion = 0.3
    save_path = os.path.join(DATA_DIR, "synthetic", "SVNH")

    data_train = torch.load("data/data-svhn/svhn-multi-digit-3x64x64_train.p")
    data_val = torch.load("data/data-svhn/svhn-multi-digit-3x64x64_val.p")
    data_test = torch.load("data/data-svhn/svhn-multi-digit-3x64x64_test.p")
    n_train = len(data_train[1])
    n_val = len(data_val[1])
    n_test = len(data_test[1])
    datasets = {"train": data_train, "val": data_val, "test": data_test}

    numbers = torch.concat([data_train[1], data_val[1], data_test[1]])
    n = numbers.shape[0]

    # max_numbers = 10000
    # numbers[numbers > max_numbers] = max_numbers

    def time_function(numbers: torch.Tensor, beta):
        # standardize
        log_numbers = np.log(numbers + 1)
        # pd.DataFrame(np.log(numbers+1)).hist(bins=100)
        BX = (log_numbers - log_numbers.float().mean()) / torch.std(log_numbers.float())
        # pd.DataFrame(np.exp(BX)).hist(bins=100)

        num_samples = BX.shape[0]
        lambda_exp_BX = (1 / 1) * np.exp(BX / 1)  # scale to mean 30 days
        lambda_exp_BX = lambda_exp_BX.flatten()

        # Generating beta samples
        U = np.random.beta(beta, beta, num_samples)

        # Exponential
        T = -np.log(U) / (lambda_exp_BX)
        return T, BX

    survival_times, BX = time_function(numbers, beta=500)
    censoring_times = np.random.uniform(0, survival_times, size=n)
    # Select proportion of the patients to be right-censored using censoring_times
    # Independent of covariates
    censoring_indices = np.random.choice(n, size=int(n * censored_proportion), replace=False)

    y_times = survival_times.float()
    y_times[censoring_indices] = torch.Tensor(censoring_times).float()[censoring_indices]

    # plt.scatter(numbers, y_times, s=1, alpha=0.1)
    # plt.scatter(numbers, survival_times, s=1, alpha=0.1)
    # ax = plt.gca()
    # ax.set_xscale('log')
    # plt.show()

    censored_events = np.zeros(n, dtype=bool)
    censored_events[censoring_indices] = True

    for s in ["train", "val", "test"]:
        idx = torch.zeros(n)
        if s == "train":
            idx[:n_train] = 1
            images = data_train[0]
        elif s == "val":
            idx[n_train : n_train + n_val] = 1
            images = data_val[0]
        else:
            idx[n_train + n_val :] = 1
            images = data_test[0]

        data = {
            "x_covar": images,
            "y_times": y_times[idx == 1],
            "censored_events": censored_events[idx == 1],
            "risk": BX[idx == 1],
            "numbers": numbers[idx == 1],
            "y_times_uncensored": survival_times[idx == 1],
        }
        name = f"{s}.pt"
        create_folder(save_path)

        torch.save(data, os.path.join(save_path, name))
        print(f"Saved SVNH dataset to: {os.path.join(save_path, name)}")

    config = {
        "name": name,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "risk_type": "beta_exponential",
        "censored_proportion": censored_proportion,
        "input_dim": (3, 64, 64),
        "output_dim": 1,
        "beta": beta,
        "setting": "synthetic",
    }

    run = wandb.init(job_type="preprocess_survSVNH", project="diffsurv", entity="cardiors")
    artifact = wandb.Artifact("SVNH", type="dataset", metadata=config)
    artifact.add_dir(save_path)
    run.log_artifact(artifact)
