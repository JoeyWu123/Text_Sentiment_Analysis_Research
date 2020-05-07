import pymongo
from pymongo import TEXT
import pandas as pd
from text_processor import*
def create_index():
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
    my_tb.create_index([('text', TEXT)])
    my_tb.create_index([('label', pymongo.ASCENDING)])
    my_tb.create_index([('sequence_no', pymongo.ASCENDING)])
def add_sequence_no():
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
    all_data=my_tb.find({})
    seq=1
    for each_row in all_data:
        id=each_row['_id']
        my_tb.update_one({'_id':id},{"$set":{"sequence_no":seq}})
        seq=seq+1
def write_sample_to_db():
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
    #uncomment following lines to reflash your database
    # if (my_tb.count_documents({})!=0):  #if there is data already in this table
    #      my_tb.delete_many({})#clear database
    data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1")
    data = data.sample(n=100100)
    count=0
    for index,row in data.iterrows():
        target=row[0]
        label=None
        if(target==4):
            label=1
        elif(target==2):
            label=0
        elif(target==0):
            label=-1
        id=row[1]
        if my_tb.count_documents({"_id":id})==0:  #notice sentiment 140 may have different rows with same ID (duplication)
            text = row[5]
            tokenized_text = processText_for_sentiment_analysis(text)
            my_tb.insert_one({"_id": id, "text": text, "tokenized_text": tokenized_text, "label":label})
            count=count+1
            if(count%1000==0):
                print(count)
        if(my_tb.count_documents({})==100000):
            break


if __name__=='__main__':
    #write_sample_to_db()
    #add_sequence_no()
    create_index()


