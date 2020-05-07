# comment the following code, if your computer doesn't have qualified Nvidia GPU which supports Cuda, or you didn't
# install tensorflow-gpu
# ---------------------------------------
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#comment/uncomment to choose CPU or GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # set the value to -1, so that the program only uses CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set the value to 0, the system will call the first GPU detected
# ---------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
from keras import regularizers
from sklearn.metrics import accuracy_score
import os
import pymongo
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)

import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
doc2vec_model= Doc2Vec.load("Doc2Vec/doc2vec_model.pickle")
def neuron_network_model(hidden_layer, hidden_unit, opt):
    model = Sequential()
    model.add(Dense(hidden_unit, input_dim=200, activation='relu'))
    for i in range(hidden_layer - 1):  # the line above already adds one hidden layer
        model.add(Dense(hidden_unit, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['accuracy'])
    return model
def grid_search():
    #prepare data
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
    raw_train_data=my_tb.find({"sequence_no": {'$nin': random_test_set_seq}})
    train_label = []
    train_token_list_matrix = []
    for each_row in raw_train_data:

        train_label.append(each_row['label'])
        train_token_list_matrix.append(each_row['tokenized_text'])
    train_dataset_vectorize = []
    for each_token_list in train_token_list_matrix:
        vector = doc2vec_model.infer_vector(each_token_list)
        train_dataset_vectorize.append(vector)
    train_data=np.array(train_dataset_vectorize[:64000])
    print(train_data)
    train_data_label=train_label[:64000]
    validate_data=train_dataset_vectorize[64000:]
    validate_data_label=train_label[64000:]
    #due to a bug in sklearn, we cannot directly call GridSearchCV to find best parameters for KerasClaasifier, otherwise,
    #"Cannot clone object <keras.wrappers.scikit_learn.KerasClassifier object at.." error will be generated.
    #https://stackoverflow.com/questions/59746974/cannot-clone-object-tensorflow-python-keras-wrappers-scikit-learn-kerasclassifi
    #Therefore, we have to write gridsearch by ourselves
    best_accuracy_score = 0
    for _hidden_layer in [3,4,5,2,1]:
        for _hidden_unit in [64,128,8,2,32,256]:
            model = KerasClassifier(build_fn=neuron_network_model, hidden_layer=_hidden_layer,
                                    hidden_unit=_hidden_unit, opt='adam', epochs=15,
                                    batch_size=32)
            model.fit(train_data, np.array(train_data_label))
            predicted_label = model.predict(np.array(validate_data))
            predicted_label = predicted_label.reshape((1, len(validate_data_label)))[0]
            score = accuracy_score(np.array(validate_data_label), predicted_label)
            print(score)
            if (score > best_accuracy_score):
                best_accuracy_score = score
                optimal_p = {"regularizer":_regularizer, "hidden_layer":_hidden_layer,
                             "hidden unit":_hidden_unit}
    print(best_accuracy_score)
    print(optimal_p)
if __name__=='__main__':
    grid_search()
    #pass