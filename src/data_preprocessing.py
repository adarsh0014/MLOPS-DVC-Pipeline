import pandas as pd # type:ignore
import numpy as np # type:ignore
from sklearn.preprocessing import LabelEncoder # type:ignore
import nltk # type: ignore
from nltk.stem.porter import PorterStemmer # type: ignore
from nltk.corpus import stopwords   # type: ignore
import string
import logging
import os
import yaml

nltk.download('stopwords')
nltk.download("punkt-tab")




# logger configuration

log_dir = 'log'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel("DEBUG")


# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

#file handler
log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(filename=log_file_path)
file_handler.setLevel("DEBUG")


formatter = logging.Formatter('%(asctime)s-%(name)s-%(lineno)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Loading the params file"""
    try:
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
            logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error("unable to load the params file %s",e)
        raise 

def transform_text(text: str):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuations, stemming
    """
    ps = PorterStemmer()
    
    #convert text to lower case
    text = text.lower()
    
    # convert text into tokens
    text = nltk.word_tokenize(text)

    # remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    # remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # stem the words
    text = [ps.stem(word) for word in text]

    return " ".join(text)


def preprocess_df(df, text_column='text', target_column='target'):
    """
    Pre process the dataframe by encoding the target column, removing dulicates and tranform the text
    """
    try:
        logger.debug('Starting the preprocessing the Dataframe')
        # droping duplicates
        df.drop_duplicates(keep='first',inplace=True)
        logger.debug('Duplicates removed')

        #encoding target column
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        logger.debug('target column get encoded')

        #transform the text
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        return df
    
    except Exception as e:
        logger.error("Unexpected error occuring during the preprocessing %s",e)


def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it and save the processed data.
    """
    try:
        params = load_params('params.yaml')
        # Fetch the data from data/raw
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data Loaded Properly")

        # Transform the data
        train_processed_data = preprocess_df(train_data,text_column=text_column, target_column=target_column)
        test_processed_data = preprocess_df(test_data,text_column=text_column, target_column=target_column)

        # store the processed data 
        data_path = os.path.join("./data",'interim')
        os.makedirs(data_path)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)

        logger.debug('Processed data saved to %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__=='__main__':
    main()












































