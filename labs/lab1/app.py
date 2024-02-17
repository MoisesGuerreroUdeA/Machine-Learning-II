import os
import pandas as pd

from flask import Flask, request
from unsupervised.dim_red.PCA import PCA

from sklearn.datasets import load_digits # MNIST dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route("/")
def LogisticRegression():
    """
    Genera los datos de entrenamiento y validación del dataset MNIST
    """
    digits = load_digits()
    digits_df = pd.DataFrame(digits['data'])
    digits_df['target'] = digits['target']
