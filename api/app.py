import os
import json
from pymongo import MongoClient
os.environ['KERAS_BACKEND'] = 'theano'
import pymongo
from flask import Flask, jsonify, request
import pandas as pd
from keras.engine.saving import model_from_json
app = Flask(__name__)


client = MongoClient('mongodb://mongodb:27017/test')
db = client.test


globals()['ml_model_name'] = None
globals()['ml_model'] = None


def get_prediction(file_name, prediction_df):
    file_path = "../ml_models/"
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    file_path = file_path + file_name

    json_file = open(file_path + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    if globals()['ml_model_name'] != file_name:
        print("new model: "+ file_name)
        globals()['ml_model'] = model_from_json(loaded_model_json)
        # load weights into new model
        globals()['ml_model'].load_weights("../ml_models/"+file_name + ".h5")
        globals()['ml_model_name'] = file_name

    return globals()['ml_model'].predict(prediction_df)


@app.route('/data', methods=['POST'])
def add_data():
    people = db.people
    request_body = json.loads(request.data)
    data = request_body.get('data')
    salary = data.get('salary', 0)
    python = data.get('python', 0)
    java = data.get('java', 0)
    cplus = data.get('c++', 0)
    js = data.get('javascript', 0)
    csharp = data.get('csharp', 0)
    rust = data.get('rust', 0)
    go = data.get('go', 0)
    php = data.get('php', 0)

    people.insert({'salary':salary,'python':python,'java':java,'c++':cplus,'javascript':js,'csharp':csharp, 'rust':rust,
                   'go':go,'php':php})

    return jsonify({'result' : 'success'})


@app.route('/list', methods=['GET'])
def get_list():
    people = db.people
    list_of_people = list(people.find({},{'_id': False}))
    return jsonify({'total':len(list_of_people),'items' : list_of_people})


@app.route('/service', methods=['POST'])
def get_service():
    data = json.loads(request.data)
    prediction_list = [0, 0, 0, 0, 0, 0, 0, 0]
    prediction_list[0] = data.get('python', 0)
    prediction_list[1] = data.get('java', 0)
    prediction_list[2] = data.get('c++', 0)
    prediction_list[3] = data.get('javascript', 0)
    prediction_list[4] = data.get('csharp', 0)
    prediction_list[5] = data.get('rust', 0)
    prediction_list[6] = data.get('go', 0)
    prediction_list[7] = data.get('php', 0)
    predict_df = pd.DataFrame(prediction_list).transpose()
    # Get latest ml-model in the database
    ml_models = db.ml_models.find({}, {'_id': False}).sort('date', pymongo.DESCENDING)
    ml_models = list(ml_models)
    if len(ml_models) == 0:
        return jsonify({'error':'no ml model found yet.'}), 200
    ml_model = ml_models[0]

    result = get_prediction(file_name=ml_model['date'], prediction_df=predict_df)

    return jsonify(({'salary':int(result[0][0])}))


if __name__ == '__main__':
    app.run(debug=True,port=8000)