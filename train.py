import tensorflow as tf
import numpy as np
from combine_datasets import PETA_DataLoader, get_label_frequencies
from models import *
import os
import datetime


if __name__ == "__main__":

    all_datasets = ["SARC3D", "GRID", "PRID", "TownCentre", "CUHK", "3DPeS", "i-LID", "CAVIAR4REID", "VIPeR", "MIT"]
    train_datasets = ["CAVIAR4REID", "GRID", "PRID", "TownCentre", "CUHK", "3DPeS", "MIT", "SARC3D", "i-LID"]
    val_datasets = ["VIPeR"]

    image_size = (96, 96)
    batch_size = 16

    features_to_keep = []
    features_file = "features_to_keep.txt"
    with open(features_file, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split("\t")
            features_to_keep.append(line[1])

    ### FAIR approach
    # train_dataloader = PETA_DataLoader(train_datasets, features_to_keep, batch_size=batch_size, image_size=image_size)
    # val_dataloader = PETA_DataLoader(val_datasets, features_to_keep, batch_size=batch_size, image_size=image_size)

    ### MIXED approach
    p_train = 0.8
    p_val = round(-(1.0-p_train), 2)
    train_dataloader = PETA_DataLoader(all_datasets, features_to_keep, batch_size=32, image_size=(96, 96), approach='mixed',
                               percent=p_train)
    val_dataloader = PETA_DataLoader(all_datasets, features_to_keep, batch_size=32, image_size=(96, 96), approach='mixed',
                                percent=p_val)

    label_frequencies = get_label_frequencies(train_dataloader, len(features_to_keep))

    for x in train_dataloader.X:
        if x in val_dataloader.X:
            raise(ValueError("Image " + x + " repeats in both dataloaders!"))

    print("Batches per train: ", len(train_dataloader))
    print("IMages per train: ", len(train_dataloader.X))
    print("Batches per val: ", len(val_dataloader))
    print("IMages per val: ", len(val_dataloader.X))

    model = model1(image_size[::-1], len(train_dataloader.features_to_keep), None)
    print(model.summary(line_length=150))

    model_save_path = "model1_weighted_medium_bce2_nolabelfreq"
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    file_name = 'epoch{epoch:02d}-loss{val_loss:.4f}.hdf5'

    tensorboard_folder = "tensorboard/" + model_save_path
    if not os.path.exists(tensorboard_folder):
        os.makedirs(os.path.join("tensorboard", model_save_path))
    log_dir = tensorboard_folder + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path + "/" + file_name, monitor='val_loss',
                                                    verbose=1, save_best_only=False, save_weights_only=True)

    model.fit_generator(train_dataloader,
                        validation_data=val_dataloader,
                        epochs=5000,
                        verbose=2,
                        callbacks=[checkpoint, tensorboard_callback],
                        )

