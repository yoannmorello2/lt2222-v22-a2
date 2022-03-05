import gzip
import random


from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import pandas as pd


import nltk
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB

def sample_lines(url,lines):
    f=gzip.open(url,'rb')
    return random.sample(list(f),lines)



def list_df_POStag(s):
    list_of_df=[]
    for sent in s:
        
        list_of_df.append(pd.DataFrame(nltk.pos_tag(WordPunctTokenizer().tokenize(str(sent).strip("b\" ,()f,                                                                                       .:;'\\").removesuffix("\\n").strip(                                                                                                ",. ;:'\"'"))),columns=['word','POS']))
        
    return list_of_df  

def processed_df(list_of_df):
    result=[]
    stop_words = stopwords.words('english')
    
    for df in list_of_df:
        df['word']=df['word'].str.lower() 
        df=df.drop(df[df['word'].isin(stop_words)].index)
        df=df.drop(df[~df['word'].str.isalpha()].index)
        df.reset_index(drop=True, inplace=True)       
        result.append(df)
    return result


def process_sentences(s):
    return processed_df(list_df_POStag(s))

def test():
    print('ca marche!')
    
def create_samples(processed_sentences, samples):
    
    #first drop the sentences with less than 5 words:
    long_enough_sentences=[df for df in processed_sentences if df.shape[0]>=5]
    
    #then we pick randomly n=samples of the left sentences
    if samples>len(long_enough_sentences):
        raise Exception("pick a smaller number of samples or a bigger list of processed_sentences")
    all_sentences=random.sample(long_enough_sentences, samples)
    
    #for each df in all_sentences, pick a random 5 elements window
    all_samp=[]
    for df in all_sentences:
        n=random.randint(0,df.shape[0]-5)
        dg=df.iloc[n:n+5,:]
        all_samp.append(dg)
    
    return all_samp



#First we transform each df in the following array: first element= last two letters of the first word
                                                    #second elt = last two letters of the second word
                                                    #third elt= last two letters of the fourth word
                                                    #fourth elt = last two letters of the 5th word
                                                    #fifth element = POS of the third word
def transform(all_samples):
    trans=[]
    for df in all_samples:
        arr=np.zeros(5,dtype=object)
        arr[0]=df['word'].iloc[0][-2:]
        arr[1]=df['word'].iloc[1][-2:]
        arr[2]=df['word'].iloc[3][-2:]
        arr[3]=df['word'].iloc[4][-2:]
        arr[4]=df['POS'].iloc[2]
        trans.append(arr)
    return trans

def create_df(all_samples):

    transformed=transform(all_samples)

    #We join them in one matrix:
    matrice=np.stack(transformed)

    #We chose a class to classify: 'NN'
    matrice[:,4]=1*(matrice[:,4]=='NN')

    enc = OneHotEncoder()
    m=enc.fit_transform(matrice[:,:4]).toarray()

    m_final=np.concatenate((m,matrice[:,4][:,None]),axis=1)
    col=[np.array(list(zip(([enc.categories_.index(arr)+1]*len(arr)),list(arr)))) for arr in enc.categories_]
    co=np.concatenate(col)
    co=np.concatenate([co,[['target','target']]])
    df=pd.DataFrame(m_final,columns=co)
    df=df.rename(columns={('target', 'target'):'target'})
    
    return df.astype(int)



def split_samples(fulldf, test_percent):
    X=fulldf.iloc[:,:-1].to_numpy()
    y=fulldf.iloc[:,-1].to_numpy()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_percent/100)
    return train_X, train_y, test_X, test_y


def train(train_X, train_y, kernel):
    clf=SVC(kernel=kernel)
    clf.fit(train_X, train_y)
    return clf


def eval_model(model, test_X, test_y):
    y_pred=model.predict(test_X)
    prec,recall,fscore,_=precision_recall_fscore_support(test_y,y_pred)
    return prec,recall,fscore


def eval_Naive_Bayes(train_X, train_y, test_X, test_y):
    gnb = GaussianNB()
    y_pred = gnb.fit(train_X, train_y).predict(test_X)
    prec,recall,fscore,_=precision_recall_fscore_support(test_y,y_pred)
    return prec,recall,fscore