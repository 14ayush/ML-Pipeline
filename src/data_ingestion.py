import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import logging

#making the lo directory
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)  #using os make the logs folder

#making the logger object 
logger=logging.getLogger('data_ingestion')  #make the object that log all the information of data_ingestion file
logger.setLevel('DEBUG')          #debug is the 1 level if we use it then we get all levels like info,warning,error,critical

#making the concole handler 

console_handler=logging.StreamHandler() #stream gandler is used to make the console handler
console_handler.setLevel('DEBUG')

#making the file handler
file_dir=os.path.join(log_dir,'data_ingestion.log') #here we are making the file data_ingestion.log and join it with the log files 
file_handler=logging.FileHandler(file_dir)
file_handler.setLevel('DEBUG')
 

#a!-----------------No need to make the two hander it is just for understanding mainly we make the file handler-----------------!


#now defining the formate 
formatr=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatr)
file_handler.setFormatter(formatr)

# now return the handlers to the object

logger.addHandler(console_handler)
logger.addHandler(file_handler)


#Now loading the data 
''''Load data from the csv file'''
def load_data(data_url:str) -> pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug('Data loaded successfully %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Unable to parse the csv file %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error %s',e)
        raise



#preprocess the data

def preproces_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        #drop the unwanted columns
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
    #rename the folders
        df.rename(columns={'v1': 'Target','v2': 'Message'},inplace=True)
        logger.debug('Prprocessing successful %s')
        return df
    except KeyError as e:
        logger.debug('These Folder not found %s',e)
        raise
    except KeyError as e:
        logger.debug('Unexpected error',e)
        raise


#saving the data 

def save_data(train_data:pd.DataFrame ,test_data:pd.DataFrame,data_path:str)->None:
    try:
        #making the directory 
        raw_data=os.path.join(data_path,'raw')
        os.makedirs(raw_data,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data,"test.csv"),index=False)
        logger.debug('data saved successfully %s',raw_data)

    except Exception as e:
        logger.debug('data not saved %s',e)
        raise


#creating the main function 
def main():
    try:
        test_size=0.2
        data_path='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
        df=load_data(data_url=data_path)
        final_data=preproces_data(df)
        train_data,test_data=train_test_split(final_data,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.debug('Failed to complete the data ingestion',e)
        print(f"Error:{e}")

if __name__=='__main__':
    main()





    



 
