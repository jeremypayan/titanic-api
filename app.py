from flask import Flask, jsonify
import pickle
from config import *
from sample.helpers import *
from sklearn.impute import SimpleImputer
import pandas as pd

app = Flask(__name__)


def get_data(path_train_data, path_test_data) :
    '''Function that imports data from the specified paths. Test and training data
       come from https://www.kaggle.com/c/titanic/submissions?group=all&page=1&pageSize=100 '''
    try :
        train_data = pd.read_csv(path_train_data, sep=',')
        test_data = pd.read_csv(path_test_data, sep=',')
    except :
        train_data = pd.read_csv(path_train_data)
        test_data = pd.read_csv(path_test_data)
    return train_data, test_data


@app.route("/titanic/predict", methods=["GET"])
def predict():
    # Gets data
    train_data, test_data = get_data('data/train.csv', 'data/test.csv')
    data = train_data[0:10]

    # Serve predictions
    loaded_model = pickle.load(open(model_path, 'rb'))
    imp = SimpleImputer(strategy="most_frequent")
    final_data = DataPreparation(data)
    final_data.filling_data(imp)
    final_data.feature_engineering()
    X_test = final_data.vectorize_test_set()
    predictions = loaded_model.predict(X_test)
    return jsonify(f"Here are {predictions} vs real values {train_data.loc[0:10, 'Survived'].values}")


if __name__ == '__main__':
    app.run(debug=True)
