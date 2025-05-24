import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import pickle
import json
import yaml
from dvclive import Live

#log function code

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

file_path=os.path.join(log_dir,'model_evaluation .log')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

#set the formate
formte=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s  ')
file_handler.setFormatter(formte)
file_handler.setLevel('DEBUG')

logger.addHandler(file_handler)

#create the function 
#loading the model


#LOAD THE PARAMS 
def load_params(params_path:str)->None:
    """Loading the parameters from the params section"""
    try:
        with open(params_path,'rb') as file:
            params=yaml.safe_load(file)
        logger.debug('Params file load successfully %s ')
        return params
    except FileNotFoundError as e:
        logger.debug('File not found %s',e)
        raise
    except yaml.YAMLError as e:
        logger.debug('YAML Error %s',e)
        raise
    except Exception as e:
        logger.debug('unexpected error %s',e)


def load_model(model_path:str)->None:
    """Loading the model """
    try:
        with open(model_path,'rb') as file:
            model=pickle.load(file)
        logger.debug('Model loaded successfullt %s',model_path)
        return model
    except Exception as e:
        logger.debug('model is not loaded successfully %s',e)
        raise 

#loading the test data

def load_data(model_path:str) -> pd.DataFrame:
    """Loading the test data for the model evaluation """
    try:
        df = pd.read_csv(model_path)
        logger.debug('Data loaded from %s', model_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
 
def model_eval(clf,X_test:np.ndarray,y_test:np.ndarray)-> dict:
    """evaluating the model s"""
    try:
        y_pred=clf.predict(X_test)
        y_pred_prob=clf.predict_proba(X_test)[:, 1]
        
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_prob)

        metric_dict = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug('Model evaluated successfully %s')
        return metric_dict
    except Exception as e:
        logger.debug('Metric evaluation unsucessfull %s',e)
        raise

#metric save

def metric_save(metrics:dict,path:str)->None:
    """Saving the metrices """
    try:
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,'w')as file:
            json.dump(metrics,file,indent=4)
        logger.debug('Metrices saved successfully %s')
    except Exception as e:
        logger.debug('unable to save the metric because metrics did not generated %s',e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed_data/test_processed.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = model_eval(clf, X_test, y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)
        
        metric_save(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()






