import os

import numpy as np
import torch.nn.functional as f
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from data.datamodules import DataModuleMLM
from definitions import DATA_DIR

if __name__ == "__main__":
    data_module = DataModuleMLM(
        token_col="phecode",
        label_col="phecode",
        token_vocab_path=os.path.join(
            DATA_DIR, "processed", "omop", "phecode_vocab_top100_105.pkl"
        ),
        label_vocab_path=os.path.join(
            DATA_DIR, "processed", "omop", "phecode_vocab_top100_105.pkl"
        ),
        age_vocab_path=os.path.join(DATA_DIR, "processed", "omop", "age_vocab_91.pkl"),
        train_data_path=os.path.join(DATA_DIR, "processed", "omop", "phe_train.parquet"),
        val_data_path=os.path.join(DATA_DIR, "processed", "omop", "phe_val.parquet"),
        test_data_path=os.path.join(DATA_DIR, "processed", "omop", "phe_test.parquet"),
        batch_size=32,
        max_len_seq=256,
        num_workers=1,
        debug=False,
        mask_prob=0.2,
    )

    train_dataloader = data_module.train_dataloader()

    x_store = []
    y_store = []
    for i, batch in enumerate(train_dataloader):
        token_idx, age_idx, position, segment, mask_labels, noise_labels = batch
        x = f.one_hot(token_idx, num_classes=105).sum(dim=1)
        mask_labels_expanded = mask_labels.view(-1)
        x = x.repeat(mask_labels.shape[1], 1)

        keep = mask_labels_expanded.view(-1) != -1
        y_store.append(mask_labels_expanded.view(-1)[keep].numpy())
        x_store.append(x[keep].numpy())
        if i > 100:
            break

    X = np.concatenate(x_store, axis=0)
    Y = np.concatenate(y_store, axis=0)

    model = LogisticRegression(penalty="l2", C=0.01, solver="lbfgs", max_iter=1000, verbose=True)
    scaler = preprocessing.StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    model.fit(X_norm, Y)
    model.score(X_norm, Y)  # 0.11 ... maybe it's just too hard a task for non sequential modeles?

    model.predict(X_norm)
