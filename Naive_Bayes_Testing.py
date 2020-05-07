import nltk
from nltk.tag import pos_tag
from nltk.corpus import twitter_samples, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import twitter_samples
from nltk import FreqDist, classify, NaiveBayesClassifier
import re,string, random
import pandas as pd
import pymongo
import numpy as np
pd.set_option('display.max_columns', 20)
from sklearn.metrics import accuracy_score,confusion_matrix

def main():

    remote_db= input("Input the IP address of your database :")
    remote_link = "mongodb://" + remote_db + ":27017/"
    my_client = pymongo.MongoClient(remote_link)
    try:
        info = my_client.server_info()  # Forces a call.
        print("Success in accessing the database")
    except ServerSelectionTimeoutError:
        print("server is down.")
        return
    my_db=my_client.get_database('Twitter_Sentiment140')
    my_tb=my_db['Sentiment140']
    np.random.seed(10)  #use seed, to make sure the result is replicable
    random_test_set_seq=np.random.choice(range(1,100001), 20000, replace=False).tolist()
    raw_test_data=my_tb.find({"sequence_no": {'$in': random_test_set_seq}})
    raw_train_data=my_tb.find({"sequence_no": {'$nin': random_test_set_seq}})

    train_label=[]
    train_token_list_matrix=[]
    test_label=[]
    test_token_list_matrix=[]
  #for Naive Bayes model in NLTK, the dataset should be processed especially
    train_dataset_for_NB=[]
    test_dataset_for_NB=[]
    for each_row in raw_train_data:
        train_label.append(each_row['label'])
        train_token_list_matrix.append(each_row['tokenized_text'])
        tweet_dict=dict([token, True] for token in each_row['tokenized_text'])
        if(each_row['label']==-1):
            train_dataset_for_NB.append((tweet_dict,"Negative"))
        else:
            train_dataset_for_NB.append((tweet_dict, "Positive"))
    for each_row in raw_test_data:
        test_label.append(each_row['label'])
        test_token_list_matrix.append(each_row['tokenized_text'])
        tweet_dict=dict([token, True] for token in each_row['tokenized_text'])
        test_dataset_for_NB.append(tweet_dict)
    NB_classifier = NaiveBayesClassifier.train(train_dataset_for_NB)

    NB_result=[]
    for each_tweet in test_dataset_for_NB:
        if NB_classifier.classify(each_tweet)=='Negative':
            NB_result.append(-1)
        else:
            NB_result.append(1)
    print("The accuracy score is: ",accuracy_score(test_label,NB_result))
    print("Confusion Matrix is: ",confusion_matrix(test_label,NB_result,labels=[-1,1]))
if __name__=='__main__':
    main()