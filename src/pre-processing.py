import pandas as pd
import numpy as np
import logging
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')



log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('pre-processing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
logger.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'pre_processing.log')
file_handler = logging.FileHandler(log_file_path)
logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.setLevel('DEBUG')

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def transform_text(text):
 
   ps = PorterStemmer()

   text = text.lower()      # converts all text to lower case

   text = nltk.word_tokenize(text)      # provides token to each word

   text = [word for word in text if word.isalnum()]         # filter outs all punctuations, special characters and single and double inverted commas

   text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]      # removes all stopwords available in english like ( the, in , are etc) and punctuaton marks

   text = [ps.stem(word) for word in text]      # makes a word shorter eg: running = run, happily = happi

   return " ".join(text)


def preprocess_df(df, text_column='text', target_column= 'target'):
   
   try:
      logger.debug('Starting the preprocessing for Dataframe')
      encoder = LabelEncoder()
      df[target_column] = encoder.fit_transform(df[target_column])
      logger.debug('target column encoded')

      df = df.drop_duplicates(keep='first')
      logger.debug('duplicates removed')

      df.loc[:, text_column] = df[text_column].apply(transform_text)
      logger.debug('text column transformed')
      return df
   
   except KeyError as e:
      logger.error('column not found %s', e)
      raise
   except Exception as e:
      logger.error('Error during text normalisation %s', e)
      raise
   

def main(text_column='text', target_column='target'):
   try:
      train_data = pd.read_csv('./data/raw/train.csv')
      test_data = pd.read_csv('./data/raw/test.csv')
      logger.debug('data loaded properly')

      train_processed_data = preprocess_df(train_data, text_column, target_column)
      test_processed_data = preprocess_df(test_data, text_column, target_column)

      data_path = os.path.join('./data', 'interim')
      os.makedirs(data_path, exist_ok=True)

      train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
      test_processed_data.to_csv(os.path.join(data_path, 'test_tranformed.csv'), index=False)

      logger.debug('preprocessed data saved %s', data_path)

   except FileNotFoundError as e:
      logger.error('file not found: %s', e)
      raise
   except pd.errors.EmptyDataError as e:
      logger.error('No data %s', e)
      raise
   except Exception as e:
      logger.error('failed to complete th e data tranformation process %s', e)
      print(f"error: {e}")

if __name__ =='__main__':
   main()
   
   

