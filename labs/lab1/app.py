import base64
import numpy as np
import pickle

from flask import Flask, request
from unsupervised.dim_red.PCA import PCA

app = Flask(__name__)

@app.route("/", methods=['POST'])
def predict():
    digit = request.get_json()['digit_img']
    digit_img = np.array([np.frombuffer(base64.b64decode(digit),dtype=np.float64)])
    print(digit_img)
    print(digit_img.shape)
    logistRegr = pickle.load(open('LogisticRegMNIST.pkl', 'rb'))
    # features = 3
    # digits_pca = PCA(features)
    # transformed_digit = digits_pca.fit_transform(digit_img)
    print(digit_img)
    print(logistRegr.predict(digit_img))
    return f"Digit prediction: {logistRegr.predict(digit_img)[0]}"
    # return "Hellow world!"

if __name__ == '__main__':
    app.run(debug = True, port = 8000)