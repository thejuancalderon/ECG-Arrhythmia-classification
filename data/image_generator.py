import math
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

LABEL = "label"
NORMAL = "N"
VENTRICULAR = "V"
SUPER_VENTRICULAR = "S"


class ECGImageGenerator(tf.keras.utils.Sequence):
    '''
    Image custom keras generator for creating images taking as input a timeseries ecg
    X: (n_samples, n_timestamps)
    y: (n_samples,)
    '''

    def __init__(self, batch_size=512, input_size=(32, 32, 3), shuffle=True, X=None, y=None):
        self.X = X.copy()
        self.y = y.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.one_hot_encoder.fit(np.array(self.y).reshape(-1, 1))

    def __getitem__(self, index):
        current_X = self.X.iloc[index * self.batch_size:(index + 1) * self.batch_size, ]
        current_y = self.y.iloc[index * self.batch_size:(index + 1) * self.batch_size, ]
        # Create images using transformations
        gaf = create_GAF_images(current_X)
        mtf = create_MTF_images(current_X)
        rp = create_RP_images(current_X)
        gaf = np.expand_dims(gaf, axis=-1)
        mtf = np.expand_dims(mtf, axis=-1)
        rp = np.expand_dims(rp, axis=-1)

        # One hot encoding of the target
        current_y = self.one_hot_encoder.transform(np.array(current_y).reshape(-1, 1))
        image = np.concatenate([gaf, mtf, rp], axis=-1)

        # Returns standard X,y
        return (image, current_y)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)


def create_GAF_images(data: pd.DataFrame = None, image_size=32):
    gaf = GramianAngularField(image_size=image_size)
    images = gaf.fit_transform(data)
    return images


def create_RP_images(data: pd.DataFrame = None, dimension=0.55):
    rp = RecurrencePlot(dimension=dimension)
    images = rp.transform(data)
    return images


def create_MTF_images(data: pd.DataFrame = None, image_size=32, n_bins=3):
    mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
    images = mtf.transform(data)
    return images


def get_generators(file='dataset.csv'):
    df = pd.read_csv(file)
    df = df.iloc[:, 1:]
    df = df.rename(columns={"0.1": LABEL})

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Under and oversampling to balance the unbalanced dataset
    under = RandomUnderSampler(sampling_strategy={NORMAL: 100000})
    X_res, y_res = under.fit_resample(X, y)
    smote = SMOTE(sampling_strategy={VENTRICULAR: 100000, SUPER_VENTRICULAR: 100000})
    X_res, y_res = smote.fit_resample(X_res, y_res)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_res, y_res, test_size=0.20, random_state=12)

    generator = ECGImageGenerator(X=Xtrain, y=ytrain, batch_size=512)
    validation_generator = ECGImageGenerator(X=Xtest, y=ytest, batch_size=512)

    return generator, validation_generator