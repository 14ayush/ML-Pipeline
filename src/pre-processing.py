#importing the necessary libraries
import pandas as pd
import nltk
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
nltk.download('punkt')

#writing the log logic
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#create the logger object 
logger=logging.getLogger('pre-processing')
logger.setLevel('DEBUG')

#now create the file handler logger
file_dir=os.path.join(log_dir,'pre-processing.log')
file_handler=logging.FileHandler(file_dir)
file_handler.setLevel('DEBUG')

formater=logging.Formatter('%(asctime)s-%(name)s-%(name)s-%(message)s')
file_handler.setFormatter(formater)
logger.addHandler(file_handler)

#preprocessing
def transform_text(text):
    """
    Transform the text into lower case remove the puncuations and the tokenization and remove the stopwords
    """
    ps=PorterStemmer()
    #convert the text into lower case
    text=text.lower()
    #tokenization of text
    text=nltk.word_tokenize(text)
    #remove the none alphanumeric 
    text=[word for word in text if word.isalpha()]
    #remove the stopwors and puncuations
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #stem the words
    text=[ps.stem(word) for word in text]
    #join the transformed test into null string
    return " ".join(text)

#defing the function for preprocess

def preprocess(df,text_column='Message',target_column='Target'):
    """
    transform the target column into 0/1 also remove the duplicates
    """
    try:
        logger.debug('preprocessing start %s')
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('Target column gets encoded %s')

        #now remove the duplicates
        df=df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed successfully %s')

        #apply text transform 
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug('text transform successfully %s')
    except KeyError as e:
        logger.debug('columns not found %s',e)
        raise
    except Exception as e:
        logger.debug('Error unable to preprocess %s',e)
        raise

def main(text_column='Message',target_column="Target"):
    """
    this is the main function used to load the data and saved it
    """
    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug('load the data successfully %s')

        #transformed data
        train_transformed_data=preprocess(train_data,text_column,target_column)
        test_transformed_data=preprocess(test_data,text_column,target_column)
        
        #store the data
        data_path=os.path.join("./data","intrim")
        os.makedirs(data_path,exist_ok=True)
        train_transformed_data.to_csv(os.path.join(data_path, "train_transform.csv"), index=False)
        test_transformed_data.to_csv(os.path.join(data_path, "test_transform.csv"), index=False)
        logger.debug('data saved successfully %s')

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
        






