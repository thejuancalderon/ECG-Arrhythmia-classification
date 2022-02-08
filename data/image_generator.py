import math
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


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

    def __init__(self, batch_size=512, input_size=(32, 32, 3), shuffle=True, X=None, y=None, oneHotEncoder=None, flag='all'):
        self.X = X.copy()
        self.y = y.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.one_hot_encoder = oneHotEncoder
        #self.one_hot_encoder = OneHotEncoder(sparse=False)
        #self.one_hot_encoder.fit(np.array(self.y).reshape(-1, 1))
        self.flag = flag

    def __getitem__(self, index):
        current_X = self.X.iloc[index * self.batch_size:(index + 1) * self.batch_size, ]
        current_y = self.y.iloc[index * self.batch_size:(index + 1) * self.batch_size, ]
        # Create images using transformations
        gaf = create_GAF_images(current_X, image_size=self.input_size[0])
        mtf = create_MTF_images(current_X, image_size=self.input_size[0])
        rp = create_RP_images(current_X, image_size=self.input_size[0])
        gaf = np.expand_dims(gaf, axis=-1)
        mtf = np.expand_dims(mtf, axis=-1)
        rp = np.expand_dims(rp, axis=-1)


        # One hot encoding of the target
        current_y = self.one_hot_encoder.transform(np.array(current_y).reshape(-1, 1))
        if(self.flag == 'gaf'):
            image = gaf
        elif(self.flag == 'mtf'):
            image = mtf
        elif(self.flag == 'rp'):
            image = rp
        else:
            image = np.concatenate([gaf, mtf, rp], axis=-1)

        # Returns standard X,y
        return (image, current_y)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)


def create_GAF_images(data: pd.DataFrame = None, image_size=32):
    gaf = GramianAngularField(image_size=image_size)
    images = gaf.fit_transform(data)
    return images


def create_RP_images(data: pd.DataFrame = None, image_size=32):
    dimension = 1 - (float(image_size) / len(data.columns))
    rp = RecurrencePlot(dimension=dimension)
    images = rp.transform(data)
    return images


def create_MTF_images(data: pd.DataFrame = None, image_size=32, n_bins=3):
    mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
    images = mtf.transform(data)
    return images

def normalize(images):
    batch_size = images.shape[0]

    for i in range(batch_size):
        image = images[i]
        min = np.min(image)
        if(min < 0):
            image = image - min

        image = image / np.max(image)
        images[i] = image

def get_generators(file='dataset.csv', batch_size=512, test_size=0.2, seed=12,
                   undersampling_cardinality=100000,
                   oversampling_cardinality=100000,
                   input_size=(32, 32, 3), flag='all'
                   ):
    df = pd.read_csv(file)
    df = df.iloc[:, 1:]
    df = df.rename(columns={"0.1": LABEL})

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # Under and oversampling to balance the unbalanced dataset
    under = RandomUnderSampler(sampling_strategy={NORMAL: undersampling_cardinality})
    X_res, y_res = under.fit_resample(Xtrain, ytrain)

    if(oversampling_cardinality>-1):
        smote = SMOTE(
            sampling_strategy={VENTRICULAR: oversampling_cardinality, SUPER_VENTRICULAR: oversampling_cardinality})
        X_res, y_res = smote.fit_resample(X_res, y_res)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(np.array(y).reshape(-1, 1))

    print(X_res.shape)

    generator = ECGImageGenerator(X=X_res, y=y_res, batch_size=batch_size, input_size=input_size, oneHotEncoder=one_hot_encoder, flag=flag)
    validation_generator = ECGImageGenerator(X=Xtest, y=ytest, batch_size=batch_size, input_size=input_size, oneHotEncoder=one_hot_encoder, flag=flag)

    return generator, validation_generator
