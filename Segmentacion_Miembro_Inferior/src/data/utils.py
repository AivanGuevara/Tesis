import os

from sklearn.model_selection import train_test_split
import pandas as pd


def move_cifar():
    """
    Move cifar-10 data from 'Descargas' to keras datasets default folder.
    """
    if not os.path.exists('~/.keras/datasets/'):
        os.makedirs('~/.keras/datasets/')

    os.rename('~/Decargas/cifar-10-python.tar.gz',
              '~/.keras/datasets/cifar-10-batches-py.tar.gz')


def load_mnist(root_folder='~/Descargas'):
    """
    Load Mnist dataset from train and test csv files.
    """
    train = pd.read_csv(os.path.join(root_folder, 'train.csv'))

    Y = train['label']
    # Drop 'label' column
    X = train.drop(labels=['label'], axis=1)
    # Reshape image in 3 dimensions (height=28px, width=28px, channel=1)
    X = X.values.reshape(-1, 28, 28, 1)

    # Split the train and the validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    return X_train, Y_train, X_test, Y_test
