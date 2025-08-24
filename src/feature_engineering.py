import pandas as pd
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
logger.setLevel('DEBUG')

log_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_path)
logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.setLevel('DEBUG')

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
        df.fillna('', inplace=True)
        logger.debug('data loaded and NANs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse CSV', e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the data', e)
        raise


def apply_tfid(train_data: pd.DataFrame, test_data: pd.DataFrame, max_feature: int) -> tuple:
    '''Here we can use the concept of  "Bag of Words": used to convert text to numbers. Very much useful in NLP 
        But now we are using TfidVectorize to conver the text to number
        Tdidvectorize and Bag of words both can be used for NLP(natural language processing)'''

    try:
        vectorize = TfidfVectorizer(max_features=max_feature)

        X_train = train_data['text'].values
        X_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values

        X_train_bow = vectorize.fit_transform(X_train)
        X_test_bow = vectorize.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('bag of data applied and data tranformed')
        return train_df, test_df

    except Exception as e:
        logger.error('Error during bag of words tranformation %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('data saved to %s', file_path)
    except Exception as e:
        logger.error('unexpected error occured while saving the file %s', e)
        raise

def main():
    try:
        # automated the max_feature using pipeline
        params = load_params(params_path='params.yaml')
        max_feature = params['feature_engineering']['max_feature']

        # hardcoded the value for manual execution
        # max_feature = 50            # it will create 50 feature(columns)

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_tranformed.csv')

        train_df, test_df = apply_tfid(train_data, test_data, max_feature=max_feature)

        save_data(train_df, os.path.join('./data', 'processed', 'train_tfid.csv'))
        save_data(test_df, os.path.join('./data', 'processed', 'test_tfid.csv'))

    except Exception as e:
        logger.error('Failed to complete feature engineering process %s', e)
        print(f"error {e}")


if __name__ == '__main__':
    main()
        
    