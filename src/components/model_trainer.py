import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
# trained_model_file_path: Path where the trained model will be saved.
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

# class handles the training and evaluation of different machine learning models.
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

# This method trains and evaluates multiple machine learning models to find the best one.
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            # X_train: includes input features (or independent variables) used to train the model
            # y_train: includes target values (or dependent variables)
            # X_test: includes input features used to test the model after it has been trained
            # y_test: includes actual target values for the testing data, used to evaluate the model’s performance
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], # All rows, and all columns except the last one
                train_array[:,-1],  # All rows, and only the last column
                test_array[:,:-1],  # All rows, and all columns except the last one
                test_array[:,-1]    # All rows, and only the last column
            )
            
            # dictionary of different models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            params={
                "Decision Tree": {
                    # criterion: The function to measure the quality of a split
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # n_estimators: The number of trees in the forest. Higher numbers can improve performance but increase computation time
                    'n_estimators': [8,16,32,64,128,256],
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                },
                "Gradient Boosting":{
                    # n_estimators: The number of boosting stages to be run
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    # learning_rate: The step size at each iteration while moving toward a minimum of a loss function
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'loss':['linear','square','exponential'],
                }
                
            }
            
            # Evaluates each model using the training and testing data, returning a report with model names and their R² scores.
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            

            
        except Exception as e:
            raise CustomException(e,sys)