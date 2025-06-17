# import sys
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# import os

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_obj

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config=DataTransformationConfig()

#     def get_data_transformer_obj(self):
#         '''
#     This function is responsible for data transformation
#     '''
#     try:
#         numerical_feat = ['writing_score', 'reading_score']
#         categorical_feat = [
#             'gender',
#             'race_ethnicity',
#             'parental_level_of_education',
#             'lunch',
#             'test_preparation_course'
#         ]

#         numerical_pipeline = Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='median')),
#             ('scaler', StandardScaler())
#         ])

#         cat_pipeline = Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('onehot', OneHotEncoder(handle_unknown='ignore')),
#             ('scaler', StandardScaler(with_mean=False))  # ✅ FIXED HERE
#         ])

#         logging.info(f"Numerical columns: {numerical_feat}")
#         logging.info(f"Categorical columns: {categorical_feat}")

#         preprocessor = ColumnTransformer([
#             ('num', numerical_pipeline, numerical_feat),
#             ('cat', cat_pipeline, categorical_feat)
#         ])
#         return preprocessor

#     except Exception as e:
#         raise CustomException(e, sys)

#     def initiate_data_transformation(self,train_path,test_path):
#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)

#             logging.info("read train and test data completed")

#             logging.info("obtaining preprocessor object")
#             preprocessing_obj=self.get_data_transformer_obj()

#             target_column_name="math_score"
#             numerical_feat=['writing_score','reading_score']

#             input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
#             target_feature_train_df = train_df[target_column_name]

#             input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
#             target_feature_test_df = test_df[target_column_name]
            


#             logging.info(
#                 f"Applying preprocessing objext on training dataframe and testing dataframe. "

#             )
#             input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

#             train_arr=np.c_[
#                 input_feature_train_arr,np.array(target_feature_train_df)
#             ]
#             test_arr=np.c_[
#                 input_feature_test_arr,np.array(target_feature_test_df)
#                 ]
#             logging.info(f"Saved preprocessing object")

#             save_obj(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj
#             )
#             return(
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path,
#             )
#         except Exception as e:
#             raise CustomException(e,sys)
            
        

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
        This function creates preprocessing pipelines for numerical and categorical features.
        """
        try:
            numerical_feat = ['writing_score', 'reading_score']
            categorical_feat = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))  # ✅ FIXED HERE
            ])

            logging.info(f"Numerical columns: {numerical_feat}")
            logging.info(f"Categorical columns: {categorical_feat}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_feat),
                    ('cat', cat_pipeline, categorical_feat)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies preprocessing to training and testing data.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_obj()
            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

