import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
logger.setLevel('DEBUG')

file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(file_path)
logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.setLevel('DEBUG')

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def load_params(params_path: str):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Params loaded %s', params_path)
        return params
    except yaml.YAMLError as e:
        logger.error('YAML error %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('file not found error %s', e)
        raise
    except Exception as e:
        logger.error('unexpected error found %s', e)
        raise
        

def load_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
        logger.debug('data loaded from %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse CSV %s', e)
        raise
    except FileExistsError as e:
        logger.error('File not found %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:                                        # checks row size is equal in both X_train and y_train [0] denotes row
            raise ValueError('The number of samples in X_train is not equal y_train')
        
        logger.debug('Initialize Randomforest model with parameters %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimator'], random_state=params['random_state'])

        logger.debug('model training started with %d samples ', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf
    except ValueError as e:
        logger.error('Valuerror during model training %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training %s', e)
        raise

def save_model(model, file_path: str) -> None:              # def save_model(clf, 'models/model.pkl')
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('file not found %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error encountered while saving the file %s', e)
        raise

def main():
    try:
        # using params to automate the pipeline
        params = load_params(params_path='params.yaml')['model_training']

        # above line can also be written as below it will return a dictionary containing n_estimator and random_state value

        # all_params = load_params(params_path='params.yaml')
        # params = all_params['model_training']

        # hardcoded value for manual execution
        # params = {'n_estimators': 25, 'random_state': 2}
        train_data = load_data('./data/processed/train_tfid.csv')
        X_train = train_data.iloc[:, :-1].values                   # All columns except the last become feature vectors.
        y_train = train_data.iloc[:, -1].values                     # The last column is the target label array.

        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('failed to complete the model building process: %s', e)
        print(f"error {e}")

if __name__ == '__main__':
    main()

