import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exeption import CustomExeption  

from src.components.data_transformation import DataTransformation, DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')

class DataIngestion:
    """
    Handles reading raw data, performing train-test split,
    and saving the processed datasets into the artifact folder.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...")
        try:
            # Read dataset
            df = pd.read_csv('/Users/jatinprakashrathore/Documents/ml_projects/notebook/data/stud.csv')
            logging.info("Successfully read dataset as DataFrame")

            # Create directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw dataset to artifacts")

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Train-test split complete: Train={train_set.shape}, Test={test_set.shape}")

            # Save datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion process completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomExeption(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transform(train_path, test_path)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    
