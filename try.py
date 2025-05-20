''''Doing the Hands on practice on the Data_ingestion part'''
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

#making the directory
log_dir='loog'
os.makedirs(log_dir,exist_ok=True)

#making the logger object
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#making the file console object
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#making the file handler so for that firstly we need to create the dir
file_dir=os.path.join(log_dir,'Data_ingestion.log')
file_handler=logging.FileHandler(file_dir)
file_handler.setLevel('DEBUG')


#now specify the formate
formatt=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatt)
file_handler.setFormatter(formatt)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


'''Load the Data'''
def load_data(data_url:str) -> pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug('data load successfully %s',data_url)
        return df
    except KeyError as e:
        logger.debug('unable to load the data',e)
        raise


def preproces(df:pd.DataFrame)-> pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
        logger.debug('preproces completed %s')
        return df
    except KeyError as e:
        logger.debug('unsuccessfull',e)
        raise


def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    #making the dir
    try:
        raw_dat=os.path.join(raw_dat,'raw')
        os.makedirs(raw_dat,exist_ok=True)
        train_data.to_csv(os.path.join(raw_dat,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_dat,"test.csv"),index=False)
        logger.debug('data saved successfully %s',raw_dat)

    except Exception as e:
        logger.debug('data not saved %s',e)
        raise

def main():
    try:
        test_size=0.2
        data_path='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
        df=load_data(data_url=data_path)
        final_data=preproces(df)
        train_data,test_data=train_test_split(final_data,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.debug('Failed to complete the data ingestion',e)
        print(f"Error:{e}")

if __name__=='__main__':
    main()





    



 





hello ji shelfdnjsfjisnbfjfknsjfnvnfnfdffnkkannkofsnko'knwskndsvnnksdonodvsnosdipweobkalfd no