import os,glob
from datetime import datetime, timedelta
from flask import Flask
from pymongo import MongoClient
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

os.environ['KERAS_BACKEND'] = 'theano'

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

client = MongoClient('mongodb://mongodb:27017/test')
db = client.test

app = Flask(__name__)
scheduler = BackgroundScheduler()


def initialize_db():
    people = db.people.find({},{'_id':False})
    people = list(people)
    if len(people) == 0:
        db.people.insert_one({"salary":5.0,"python":0,"java":0,"c++":0,"javascript":0,"csharp":0,"rust":0,"go":0,"php":1})
        db.people.insert_one({"salary": 5.5, "python": 0, "java": 0, "c++": 0, "javascript": 1, "csharp": 0, "rust": 0, "go": 0, "php": 1})
        db.people.insert_one({"salary": 6, "python": 0, "java": 0, "c++": 0, "javascript": 1, "csharp": 1, "rust": 1, "go": 1, "php": 0})
        db.people.insert_one({"salary": 6.5, "python": 0, "java": 1, "c++": 0, "javascript": 1, "csharp": 1, "rust": 0, "go": 0, "php": 0})
        db.people.insert_one({"salary": 7, "python": 1, "java": 0, "c++": 1, "javascript": 1, "csharp": 0, "rust": 1, "go": 0, "php": 0})
        db.people.insert_one({"salary": 7.5, "python": 1, "java": 1, "c++": 1, "javascript": 0, "csharp": 1, "rust": 0, "go": 0, "php": 0})

    return True


def train_task():
    # Data preparation
    people = db.people.find({},{'_id':False})
    people = list(people)

    data = []
    for person in people:
        train_list = [0,0,0,0,0,0,0,0,0]
        train_list[0] = person.get('salary', 0)
        train_list[1] = person.get('python', 0)
        train_list[2] = person.get('java', 0)
        train_list[3] = person.get('c++', 0)
        train_list[4] = person.get('javascript', 0)
        train_list[5] = person.get('c#', 0)
        train_list[6] = person.get('rust', 0)
        train_list[7] = person.get('go', 0)
        train_list[8] = person.get('php', 0)
        data.append(train_list)

    # %%
    train_df = pd.DataFrame(data)
    train_df.columns = ['salary','python','java','c++','javascript','csharp','rust','go','php']

    train_X = train_df.drop(columns=['salary'])
    # check that the target variable has been removed
    # train_X.head()
    # create a dataframe with only the target column
    train_y = train_df[['salary']]
    # %%
    model = Sequential()
    # get number of columns in training data
    n_cols = train_X.shape[1]
    # add model layers

    model.add(Dense(8, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping_monitor = EarlyStopping(patience=3)

    # train model
    model.fit(train_X, train_y, validation_split=0.2, epochs=500, callbacks=[early_stopping_monitor])
    model_json = model.to_json()

    # %%
    now = datetime.now()
    one_day_ago = now - timedelta(days=1)
    one_day_ago = one_day_ago.isoformat()
    now = now.isoformat()
    db.ml_models.insert({"date":now})

    file_path = "../ml_models/"
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    file_name = file_path + str(now)

    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(file_name + ".h5")

    for filename in glob.glob(file_path + one_day_ago + "*"):
        os.remove(filename)

    return True


if __name__ == '__main__':
    initialize_db()
    train_task()
    scheduler.add_job(func=train_task, trigger="interval", seconds=300)
    scheduler.start()
    app.run(debug=True,port=5001)
