import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import yaml
#creating the log
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#create the logger object
logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

#creting the file handler loger 
file_path=os.path.join(log_dir,'feature_engineering.log')
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

#define the function

def load_data(data_url:str) ->pd.DataFrame:
    """
    Load the transformed data 

    """
    try:
        df=pd.read_csv(data_url)
        df. fillna('',inplace=True)
        logger.debug('Data load successfully %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.debug('Data canot be parse %s',e)
        raise
    except Exception as e:
        logger.debug('Unable to execute %s',e)
        raise

# now applying the feature selection 
def feat(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple:
    """
    apply the vectorixation and try to find out the maximun features 
    """
    try:
        vectorizer=TfidfVectorizer(max_features=max_features)

        X_train=train_data['Message'].values
        y_train=train_data['Target'].values
        X_test=test_data['Message'].values
        y_test=test_data['Target'].values

        X_train_row=vectorizer.fit_transform(X_train)
        X_test_row=vectorizer.fit_transform(X_test)

        train_df = pd.DataFrame(X_train_row.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_row.toarray())
        test_df['label'] = y_test

        logger.debug('Feature applied and data transformed %s')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


#save the data
def save_data(df:pd.DataFrame,data_path:str) ->None:
    """
    save the featured select data into csv
    """
    try:
        os.makedirs(os.path.dirname(data_path),exist_ok=True)
        df.to_csv(data_path,index=False)
        logger.debug('Data Saved Succesfully %s',data_path)
    except Exception as e:
        logger.debug('There is an error to save the data %s',e)
        raise 

def main():
    try:
        params=load_params(params_path='params.yaml')
        max_features=params['feature_engineering']['max_features']
        #max_features=50
        train_data=load_data('./data/intrim/train_transform.csv')
        test_data=load_data('./data/intrim/test_transform.csv')
        train_df,test_df=feat(train_data,test_data,max_features)

        save_data(train_df,os.path.join("./data","processed_data","train_processed.csv"))
        save_data(test_df,os.path.join("./data","processed_data","test_processed.csv"))

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()



