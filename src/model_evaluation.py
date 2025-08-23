import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_auc_score
import pickle
import json

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
logger.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.py')
file_handler = logging.FileHandler(log_file_path)
logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.setLevel('DEBUG')

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded successfully %s', file_path)
        return model
    except FileNotFoundError as e:
        logger.error('file not found %s', e)
        raise
    except Exception as e:
        logger.error('unexpected error encountered %s', e)
        raise
        

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Parse error encountered during loading %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('file not found %s', e)
        raise
    except Exception as e:
        logger.error('unexpected error encountered %s', e)
        raise


def evalute_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('model evaluation metrices calculated')
        return metrics_dict
    except Exception as e:
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('metrices saved to %s', file_path)
    except Exception as e:
        logger.error('error occured while saving the metrices %s', e)
        raise

def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfid.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evalute_model(clf, X_test, y_test)

        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('failed to complete model evaluation process %s', e)
        print(f"error {e}")

if __name__ == "__main__":
    main()


