import os
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yaml

#creating the log
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#create the logger object
logger=logging.getLogger('model_selection')
logger.setLevel('DEBUG')

#creting the file handler loger 
file_path=os.path.join(log_dir,'model_selection.log')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

#set the formate
formte=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s  ')
file_handler.setFormatter(formte)
file_handler.setLevel('DEBUG')

logger.addHandler(file_handler)


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


#load the data 
def load_data(data_url:str)->None:
    """Loading the processed data """
    try:
        df=pd.read_csv(data_url)
        logger.debug('Data loaded successfully %s',data_url)
        return df
    except Exception as e:
        logger.debug('File not present at the given location %s',e)
        raise

#apply the modele classification 
def model_build(X_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    """Here we are applying the random forest to the processed data """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('The x_train and y train mush have same no of samples')
        logger.debug('initilizing the parameters %s',params)

        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug('Model training initilized %s',X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug('model train successful')
        return clf
    except Exception as e :
        logger.debug('Model training fails %s',e)
        raise

def save_moedel(model,data_url:str)->None:
    """saving the model into pkl file"""
    try:
        os.makedirs(os.path.dirname(data_url),exist_ok=True)

        with open(data_url,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved successfully %s ',data_url)
    except Exception as e:
        logger.error('file not found %s',e)
        raise

def main():
    """initilize the parameters and data path"""
    try:
        params=load_params('params.yaml')['model_selection']
        
        #params={'n_estimators':25,'random_state':2}
        train_data=load_data('./data/processed_data/train_processed.csv')
        x_train=train_data.iloc[:, :-1].values
        y_train=train_data.iloc[:, -1].values
        clf=model_build(x_train,y_train,params)

        #save the model 
        model_save_path='models/model.pkl'
        save_moedel(clf,model_save_path)

        logger.debug('Model saved successfully %s')
    except Exception as e:
        logger.debug('unable to procede %s')
        raise

if __name__=='__main__':
    main()
        


