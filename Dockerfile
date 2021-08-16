FROM jupyter/scipy-notebook

RUN python -m pip install joblib
RUN python -m pip install tensorflow
RUN python -m pip install opencv-python
RUN python -m pip install pandas
RUN python -m pip install tqdm
RUN python -m pip install imutils

COPY PETA_dataset ./PETA_dataset
COPY best_model.hdf5 ./best_model.hdf5

COPY models.py ./models.py
COPY predict.py ./predict.py
COPY combine_datasets.py ./combine_datasets.py
COPY features_to_keep.txt ./features_to_keep.txt


