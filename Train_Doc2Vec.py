import multiprocessing
import pymongo
cores = multiprocessing.cpu_count()
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from text_processor import*
def main():
    db_add = "localhost"
    link = "mongodb://" + db_add + ":27017/"
    my_client = pymongo.MongoClient(link)
    try:
        info = my_client.server_info()  # Forces a call.
        print("Success in accessing the database")
    except ServerSelectionTimeoutError:
        print("server is down.")
        return
    all_tokenized_text = []
    print("Begin loading texts from Sentiment 140")
    #Get data of sentiment 140
    my_db = my_client.get_database('Twitter_Sentiment140')
    my_tb = my_db["Sentiment140"]
    rows = my_tb.find({}, {"tokenized_text": 1})
    for each_row in rows:
        all_tokenized_text.append(each_row['tokenized_text'])
    print("Finish loading texts from Sentiment 140")
    #get data of IMDb movie reviews
    print("Begin loading texts from IMDb movie reviews")
    my_db = my_client.get_database('IMDb')
    my_tb = my_db["movie_reviews"]
    rows = my_tb.find({}, {"review_titles":1,"comment": 1})
    count=0
    for each_row in rows:
        raw_text=each_row["review_titles"]+" "+each_row['comment']
        clean_text=processText_for_sentiment_analysis(raw_text)
        all_tokenized_text.append(clean_text)
        count=count+1
        if(count%1000==0):
            print("Already loaded: ",count)
    print("Finish loading texts from IMDb movie reviews")
    print("Begin training Doc2Vec model")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_tokenized_text)]
    doc2vec_model = Doc2Vec(documents, vector_size=200, min_count=1, workers=cores)
    doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=20)
    doc2vec_model.save('doc2vec_model.pickle')
    print("done")
if __name__=='__main__':
    main()
