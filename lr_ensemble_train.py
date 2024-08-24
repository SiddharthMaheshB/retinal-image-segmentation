import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

from segnet_model import build_segnet
from unet_model import build_unet

h = 512
w = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    ori_x = x
    x = x / 255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ori_x = x
    x = x / 255.0
    x = x.astype(np.int32)
    return ori_x, x

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((h, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def ensemble_predict(x, unet_model, segnet_model, logistic_regression_model):
    unet_pred = unet_model.predict(np.expand_dims(x, axis=0))[0]
    segnet_pred = segnet_model.predict(np.expand_dims(x, axis=0))[0]
    
    unet_pred = unet_pred.flatten().reshape(-1, 1)
    segnet_pred = segnet_pred.flatten().reshape(-1, 1)
    
    ensemble_input = np.concatenate([unet_pred, segnet_pred], axis=1)
    ensemble_pred = logistic_regression_model.predict_proba(ensemble_input)[:, 1]
    
    ensemble_pred = ensemble_pred.reshape((h, w))
    ensemble_pred = (ensemble_pred > 0.5).astype(np.float32)
    
    return ensemble_pred

def train_logistic_regression_model(train_x, train_y, unet_model, segnet_model, save_path):
    X_train = []
    y_train = []

    for x, y in tqdm(zip(train_x, train_y), total=len(train_x)):
        _, x = read_image(x)
        _, y = read_mask(y)

        unet_pred = unet_model.predict(np.expand_dims(x, axis=0))[0]
        segnet_pred = segnet_model.predict(np.expand_dims(x, axis=0))[0]
        
        unet_pred = unet_pred.flatten().reshape(-1, 1)
        segnet_pred = segnet_pred.flatten().reshape(-1, 1)
        
        ensemble_input = np.concatenate([unet_pred, segnet_pred], axis=1)
        X_train.append(ensemble_input)
        y_train.append(y.flatten())

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train)

    joblib.dump(logistic_regression_model, save_path)
    print(f"Logistic Regression model saved to {save_path}")

    return logistic_regression_model

def evaluate_ensemble(test_x, test_y, unet_model, segnet_model, logistic_regression_model):
    SCORE = []

    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.basename(x).split(".")[0]

        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        y_pred = ensemble_predict(x, unet_model, segnet_model, logistic_regression_model)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        save_image_path = f"results/{name}.png"
        save_results(ori_x, ori_y, y_pred, save_image_path)

        y = y.flatten()
        y_pred = y_pred.flatten()

        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="macro")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="macro")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="macro")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="macro")
        SCORE.append([acc_value, f1_value, jac_value, recall_value, precision_value])
    
    score = np.mean(SCORE, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/lr_ensemble_scores.csv")

if __name__ == "__main__":
    create_dir("results")

    input_shape = (512, 512, 3)
    
    unet_model = build_unet(input_shape)
    unet_model.load_weights('files/unet_model.h5')
    
    segnet_model = build_segnet(input_shape)
    segnet_model.load_weights('files/segnet_model.keras')

    dataset_path = "new_data/train"
    train_x, train_y = load_data(dataset_path)
    
    lr_path = "files/lr_model.joblib"
    if os.path.exists(lr_path):
        logistic_regression_model = joblib.load(lr_path)
        print(f"Loaded existing logistic regression modelf rom {lr_path}")
    else:
        logistic_regression_model = train_logistic_regression_model(train_x, train_y, unet_model, segnet_model, lr_path)

    dataset_path = "new_data/test"
    test_x, test_y = load_data(dataset_path)

    evaluate_ensemble(test_x, test_y, unet_model, segnet_model, logistic_regression_model)

