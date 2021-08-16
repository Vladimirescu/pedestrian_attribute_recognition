import numpy as np
import os
import tensorflow as tf
import cv2
import imutils


def image_file_to_name(dataset_name):
    """
    :param dataset_name: PETA sub-dataset name
    :return: a function mapping image file name to corresponding name from .txt label file
    """

    if dataset_name == "3DPeS":
        return lambda x: x.split("_")[0]

    elif dataset_name == "CAVIAR4REID":
        return lambda x: x.split("_")[0]

    elif dataset_name == "CUHK":
        return lambda x: x

    elif dataset_name == "GRID":
        return lambda x: x.split("_")[0]

    elif dataset_name == "i-LID":
        return lambda x: x.split("_")[0]

    elif dataset_name == "MIT":
        return lambda x: x.split("_")[0]

    elif dataset_name == "PRID":
        return lambda x: x.split("_")[0]

    elif dataset_name == "SARC3D":
        return lambda x: x.split("_")[0]

    elif dataset_name == "TownCentre":
        return lambda x: x.split("_")[0]

    elif dataset_name == "VIPeR":
        return lambda x: x.split("_")[0]

    else:
        raise ValueError("Dataset " + dataset_name + " does not exist!")


class PETA_DataLoader(tf.keras.utils.Sequence):

    def __init__(self, data_sets, features_to_keep, batch_size=16, image_size=(96, 96), approach='fair', percent=None):
        """
        :param data_sets: if approach == 'fair' => train & validation loader will have different data_sets
                          if approach == 'mixed' => train & validation loader will contain images from all data_sets
        :param features_to_keep: list of strings for the attributes to keep
        :param batch_size: -
        :param image_size: size for all the images to be reshaped
        :param approach: 'fair' or 'mixed'
        :param percent: given only if approach == 'mixed' => percentage of images to hold from all data_sets
            percent > 0 => first 'percent' of images from each dataset (order of files given by os.listdir)
            percent < 0 => last 'percent' of images from each dataset (order of files given by os.listdir)
        """
        self.data_sets = data_sets
        self.batch_size = batch_size
        self.image_size = image_size
        self.features_to_keep = np.array(features_to_keep)
        self.percent = percent
        if approach == 'fair':
            self.X, self.y = self.construct_XY()
        elif approach == 'mixed':
            assert percent is not None, "Attribute 'percent' can't be None"
            self.X, self.y = self.construct_XY_mixed()

        self.flip_prob = 0.3
        self.rotate_prob = 0.3
        self.smooth_prob = 0.5

        assert len(self.X) == len(self.y), "Lengths of X & y don't match!"

    def construct_XY(self):
        """
        :return: list of image paths (X) obtained from all self.data_sets & list of string attributes
        """
        X = []
        y = []

        for d in self.data_sets:
            files = os.listdir("PETA_dataset/" + d + "/archive/")
            idx_label = files.index("Label.txt")
            label_file = files[idx_label]

            del files[idx_label]

            map_func = image_file_to_name(d)
            image_names = list(map(map_func, files))
            lines = []
            with open("PETA_dataset/" + d + "/archive/" + label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    line = line.split(" ")
                    lines.append(line)

            labels_dict = {x[0]: x[1:] for x in lines}

            for i in range(len(files)):
                X.append("PETA_dataset/" + d + "/archive/" + files[i])
                y.append(labels_dict[image_names[i]])

        return X, y

    def construct_XY_mixed(self):
        """
        :return: list of image paths (X) obtained from all datasets combined & list of string attributes
        """
        X = []
        y = []

        for d in self.data_sets:
            files = os.listdir("PETA_dataset/" + d + "/archive/")
            idx_label = files.index("Label.txt")
            label_file = files[idx_label]

            del files[idx_label]

            if self.percent > 0:
                n_files = int(np.ceil(abs(self.percent) * len(files)))
                files = files[:n_files]
            elif self.percent < 0:
                n_files = int(np.floor(abs(self.percent) * len(files)))
                files = files[-n_files:]

            map_func = image_file_to_name(d)
            image_names = list(map(map_func, files))
            lines = []
            with open("PETA_dataset/" + d + "/archive/" + label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    line = line.split(" ")
                    if line[0] in image_names:
                        lines.append(line)

            labels_dict = {x[0]: x[1:] for x in lines}

            for i in range(len(files)):
                X.append("PETA_dataset/" + d + "/archive/" + files[i])
                y.append(labels_dict[image_names[i]])

        return X, y

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, idx, *args, **kwargs):
        X = self.X[idx * self.batch_size: (idx + 1) * self.batch_size]
        y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        images = []
        labels = np.zeros((len(X), len(self.features_to_keep)))

        for i in range(len(X)):
            im = cv2.imread(X[i], cv2.IMREAD_COLOR)
            if self.image_size is not None:
                im = cv2.resize(im, self.image_size, interpolation=cv2.INTER_LINEAR)
            if np.max(im) > 1:
                im = im.astype(np.float32) / 255

            p_flip = np.random.random()
            if p_flip < self.flip_prob:
                im = cv2.flip(im, 1)

            p_rotate = np.random.random()
            if p_rotate < self.rotate_prob:
                angle = np.random.uniform(low=-15, high=15)
                im = imutils.rotate(im, angle=angle)

            p_smooth = np.random.random()
            if p_smooth < self.smooth_prob:
                im = cv2.GaussianBlur(im, (5, 5), 0)

            images.append(im)

        for i in range(len(y)):
            for j in range(len(self.features_to_keep)):
                if self.features_to_keep[j] in y[i]:
                    labels[i, j] = 1

        images = np.asanyarray(images)
        labels = np.asanyarray(labels)
        return images, labels


def get_label_frequencies(data_loader, n_features):
    frequencies = np.zeros(n_features)
    for X, y in data_loader:
        frequencies += tf.reduce_sum(y, axis=0).numpy()

    r = 1 - frequencies / len(data_loader.X)
    weights = np.exp(r)

    return tf.cast(weights, tf.float32)


# if __name__ == "__main__":
#
#     dataset_folder = "PETA_dataset"
#     data_sets = os.listdir(dataset_folder)
#     data_sets = [dataset_folder + "/" + x + "/archive" for x in data_sets]
#
#     for d in data_sets:
#         print(d, len(os.listdir(d)), " images")
#
#     features_to_keep = []
#     features_file = "features_to_keep.txt"
#     with open(features_file, "r") as f:
#         for line in f:
#             line = line.strip()
#             line = line.split("\t")
#             features_to_keep.append(line[1])
#
#     train_datasets = ["CAVIAR4REID", "SARC3D", "GRID", "PRID", "TownCentre", "CUHK", "3DPeS"]
#     val_datasets = ["i-LID", "VIPeR", "MIT"]
#
#     dummy = ["SARC3D", "GRID", "PRID", "TownCentre", "CUHK", "3DPeS", "i-LID", "CAVIAR4REID", "VIPeR", "MIT"]
#     dummy_dl = PETA_DataLoader(dummy, features_to_keep, batch_size=16, image_size=(96, 96), approach='mixed', percent=0.8)
#     dummy_dl2 = PETA_DataLoader(dummy, features_to_keep, batch_size=16, image_size=(96, 96), approach='mixed', percent=-0.2)
#     print(len(dummy_dl.X), len(dummy_dl.y))
#     print(len(dummy_dl2.X), len(dummy_dl2.y))
#
#     # for x in dummy_dl.X:
#     #     if x in dummy_dl2.X:
#     #         print("dubluri")
#
#     # X, y = dummy_dl[88]
#
#     # print(X.shape, y.shape)
#     # for i in range(X.shape[0]):
#     #     print(y[i])
#     #     rot = imutils.rotate(X[i], angle=-15)
#     #     flip = cv2.flip(X[i], 1)
#     #     img = cv2.GaussianBlur(X[i], (5, 5), 0)
#     #     cv2.imshow("im", (X[i] * 255).astype(np.uint8))
#     #     # cv2.imshow("im_gauss", (img * 255).astype(np.uint8))
#     #     # cv2.imshow("im_rot", (rot * 255).astype(np.uint8))
#     #     # cv2.imshow("im_flip", (flip * 255).astype(np.uint8))
#     #     cv2.waitKey(0)
#     #
#     # print(X.shape, y.shape)
#
#     print(get_label_frequencies(dummy_dl, len(features_to_keep)))
