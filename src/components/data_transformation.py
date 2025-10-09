import sys
import os
from dataclasses import dataclass
import numpy as np
from src.utils import save_object
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exeption import CustomExeption
from src.logger import logging



@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()
    
    def get_data_tranformed_obj(self):

        '''
        This function si responsible for data trnasformation
        
        '''

        try:
            numerical_feature=["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipline=Pipeline(
                steps=[
                    ("inputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_feature}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline",num_pipline,numerical_feature),
                    ("cat_pipline",cat_pipline,categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomExeption(e,sys)
        
    def initiate_data_transform(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("reading the train and test data")
            
            logging.info("obtaining preprocessing object")
            
            preprocessor_obj=self.get_data_tranformed_obj()
            
            target_col="math_score"
            numerical_col=["writing_score", "reading_score"]
            input_features_train_df=train_df.drop(columns=['math_score'],axis=1)
            output_features_train_df=train_df[target_col]
            
            
            input_features_test_df=test_df.drop(columns=['math_score'],axis=1)
            output_features_test_df=test_df[target_col]

            logging.info("applying the preprocessing in the train data set")
            
            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessor_obj.transform(input_features_test_df)
            
            train_arr = np.concatenate([input_features_train_arr, np.array(output_features_train_df).reshape(-1, 1)], axis=1)
            test_arr = np.concatenate([input_features_test_df, np.array(output_features_test_df).reshape(-1, 1)], axis=1)

            
            logging.info("saved preprocessing object")

            save_object(

                file_path=self.data_transformation.preprocessor_obj_path,
                obj=preprocessor_obj

            )

            return (
                train_arr,test_arr,self.data_transformation.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomExeption(e,sys)
        
