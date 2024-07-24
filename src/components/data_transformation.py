import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
# StandardScalar is used for normalisation, makes mean of num data 0

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# Defines a data class for storing the path to the preprocessor object file.
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

# DataTransformation: A class that performs data transformation tasks.
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


# This method creates a preprocessing object that handles the data transformation process. It sets up pipelines for both numerical and categorical features.
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
# SimpleImputer: Fills missing values with the median of the column.
# StandardScaler: Scales the numerical features to have zero mean and unit variance.
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

# SimpleImputer: Fills missing values with the most frequent value in the column.
# OneHotEncoder: Converts categorical variables into a form that could be provided to ML algorithms to do a better job in prediction.
# StandardScaler: Scales the categorical features, but with_mean=False ensures that the mean is not subtracted (necessary for sparse data).
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

# ColumnTransformer: Applies different preprocessing steps to numerical and categorical columns.
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

# initiate_data_transformation: this method reads the data, applies transformation and saves the preprocessed data.
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

# Target Column: The column that needs to be predicted (e.g., math_score).
# Input Features: All columns except the target column.
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

# Predefined in included in skikit-learn - fit_transform: used on training data, learns the parameters and apply them to the transformation.
# Predefined in included in skikit-learn - transform: used on testing data, ensures new data is transformed consistently using learned parameters from training set.
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

# Combines the transformed input features with the target column for both train and test datasets.
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)