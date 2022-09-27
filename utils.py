import numpy as np
import pandas as pd
import tensorflow as tf
import json


def load_model():
    model = tf.keras.models.load_model("transformer_N10")
    return model


def load_seed_products():
    with open("data/dinner_dictionary.json") as json_file:
        data = json.load(json_file)
    return data

def load_predictions():
    with open("data/predictions.json") as json_file:
        data = json.load(json_file)
    for k, v in data.items():
        data[k] = np.array(v)
    return data


def load_products_df():
    return pd.read_csv("data/products.csv")


def load_category_mapping():
    return pd.read_csv("data/categories.csv")


def create_idx_to_id_mapping():
    product_df = load_products_df()
    mapping = dict(zip(product_df.product_idx, product_df.product_id))
    return mapping


def create_id_to_idx_mapping():
    product_df = load_products_df()
    mapping = dict(zip(product_df.product_id, product_df.product_idx))
    return mapping


def create_id_to_name_mapping():
    product_df = load_products_df()
    mapping = dict(zip(product_df.product_id, product_df["name"]))
    return mapping


def create_idx_to_name_mapping():
    product_df = load_products_df()
    mapping = dict(zip(product_df.product_idx, product_df["name"]))
    return mapping

def create_idx_to_category_two_mapping():
    category_df = load_category_mapping()
    mapping = dict(
        zip(category_df.product_idx, category_df.product_category_name_level_two)
    )
    return mapping


def predict_next(model: tf.keras.Model, inputs: list, k: int = 1):
    inputs = np.array(inputs).reshape((1, -1))
    pred = model(inputs)
    pred = tf.nn.softmax(tf.reshape(pred[:, -1, :], (-1))).numpy()

    sorted_preds = np.argsort(-pred)

    idx = sorted_preds[:k]
    scores = pred[idx]

    return idx, scores