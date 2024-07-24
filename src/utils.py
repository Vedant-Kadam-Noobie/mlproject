import os
import sys

import numpy as np
import pandas as pd

# pickle/dill are used to serialize and deserialize Python objects
# Usage: Commonly used to save machine learning models, preprocessors, or any Python objects to a file and load them later.
import dill

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