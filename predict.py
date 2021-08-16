import tensorflow as tf
from combine_datasets import PETA_DataLoader, get_label_frequencies
from models import model1
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_class_metrics(model, data_loader, feature_names, file_name=None):
    tp = np.zeros(len(data_loader.features_to_keep))
    tn = np.zeros(len(data_loader.features_to_keep))
    fp = np.zeros(len(data_loader.features_to_keep))
    fn = np.zeros(len(data_loader.features_to_keep))
    count = np.zeros(len(data_loader.features_to_keep))

    for X, y in tqdm(data_loader):
        y_pred = model(X)
        y_pred = tf.cast(y_pred > 0.5, tf.float32).numpy()
        # m_accuracies = m_accuracies + np.sum(np.array(y == y_pred, dtype=int), axis=0)
        # m_precisions = m_precisions + np.sum(np.array(y * y_pred == 1), axis=0)
        tp = tp + np.sum(np.array(y * y_pred == 1), axis=0)
        tn = tn + np.sum(np.array((1-y) * (1-y_pred)) == 1, axis=0)
        fp = fp + np.sum(np.array((1-y) * y_pred) == 1, axis=0)
        fn = fn + np.sum(np.array((1-y_pred) * y) == 1, axis=0)
        count += np.sum(y, axis=0)

    m_accuracies = (tp + tn) / (tp + tn + fp + fn)
    m_precisions = tp / (tp + fp)
    m_recall = tp / (tp + fn)
    count_freq = count / len(data_loader.X)
    df = pd.DataFrame(list(zip(m_accuracies, m_precisions, m_recall, count_freq)), columns=["Class Accuracies",
                                                                                  "Class precisions",
                                                                                  "Class recalls",
                                                                                  "Class frequencies"],
                      index=feature_names)

    print("Saving metrics..")
    df.to_csv(file_name)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


if __name__ == "__main__":

    all_datasets = ["SARC3D", "GRID", "PRID", "TownCentre", "CUHK", "3DPeS", "i-LID", "CAVIAR4REID", "VIPeR", "MIT"]
    train_datasets = ["SARC3D", "GRID", "PRID", "TownCentre", "CUHK", "3DPeS"]
    val_datasets = ["i-LID", "CAVIAR4REID", "VIPeR", "MIT"]
    image_size = (96, 96)
    batch_size = 16

    features_to_keep = []
    features_file = "features_to_keep.txt"
    with open(features_file, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split("\t")
            features_to_keep.append(line[1])

    # FAIR approach
    # train_dataloader = PETA_DataLoader(train_datasets, features_to_keep, batch_size=batch_size, image_size=image_size)
    # val_dataloader = PETA_DataLoader(val_datasets, features_to_keep, batch_size=batch_size, image_size=image_size)

    ### MIXED approach
    p_train = 0.7
    p_val = round(-(1.0 - p_train), 2)
    train_dataloader = PETA_DataLoader(all_datasets, features_to_keep, batch_size=16, image_size=(96, 96),
                                       approach='mixed',
                                       percent=p_train)
    val_dataloader = PETA_DataLoader(all_datasets, features_to_keep, batch_size=16, image_size=(96, 96),
                                     approach='mixed',
                                     percent=p_val)

    label_frequencies = get_label_frequencies(train_dataloader, len(features_to_keep))

    model = model1(image_size[::-1], len(train_dataloader.features_to_keep), None)
    print(model.summary(line_length=150))

    model_path = "best_model.hdf5"
    model.load_weights(model_path)

    feature_names = train_dataloader.features_to_keep
    print("Train accuracies:\n")
    get_class_metrics(model, train_dataloader, feature_names, file_name="train_metrics")
    print("Val accuracies:\n")
    get_class_metrics(model, val_dataloader, feature_names, file_name="train_metrics")


