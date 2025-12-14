import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import json
import yaml
from dvclive import Live


# logging configuration

log_dir = 'log'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# file handler
log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s-%(name)s-%(lineno)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_model(model_path:str):
    """Load the model"""

    try:
        logger.debug("start model loading")
        with open(model_path,'rb') as f:
            model = pickle.load(f)
        logger.debug("model loaded successfully from %s",model_path)
        return model

    except FileNotFoundError:
        logger.error('File not found: %s', model_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path:str) -> pd.DataFrame:
    """Load data from a csv file"""

    try:
        logger.debug("start loading the data")
        df = pd.read_csv(file_path)
        logger.debug("data loaded successfully from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found on specified path %s",file_path)
        raise
    except Exception as e:
        logger.error("Unable to load the data %s",e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,-1]

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics:dict, file_path:str)->None:
    """Save the metrics on the mentioned path"""

    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            json.dump(metrics,file)
        logger.debug("Metrics saved to %s",file_path)

    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise



def main():
    
    try:
        params = load_params("params.yaml")

        model_path = './models/model.pkl'
        file_path = './data/processed/test_tfidf.csv'

        model = load_model(model_path)

        test_data = load_data(file_path)

        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evaluate_model(model,X_test,y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)
        save_metrics(metrics,'reports/metrics.json')

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()


































































