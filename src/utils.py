import os
import sys

import numpy as np 
import pandas as pd

# pickle/dill are used to serialize and deserialize Python objects
# Usage: Commonly used to save machine learning models, preprocessors, or any Python objects to a file and load them later.
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# save_object: saving a preprocessor object, which includes transformations for both numerical and categorical data, ensuring the same transformations can be applied consistently to new data.
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Train and evaluate multiple machine learning models, applying hyperparameter tuning, and return their performance scores
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        # Loop through each model
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            # Perform hyperparameter tuning & Cross-validation using GridSearchCV
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)