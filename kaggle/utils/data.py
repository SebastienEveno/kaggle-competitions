from typing import Tuple
import pandas as pd
import os

from sklearn import preprocessing

class DatasetManager:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.path = os.path.join("datasets", f"{dataset_name}/data")
        self.df_train = pd.read_csv(os.path.join(self.path, "train.csv"))
        self.df_test = pd.read_csv(os.path.join(self.path, "test.csv"))
    
    def load_data(self) -> Tuple:
        return self.df_train, self.df_test
    
    def save_data(self, 
        df: pd.DataFrame, 
        file_name: str, 
        with_index: bool=False) -> None:

        df.to_csv(os.path.join(self.path, f"{file_name}.csv"), index=with_index)

    def preprocess_data(self, 
        df: pd.DataFrame,
        is_training_data: bool=True,
        drop_na: bool=False, 
        reshuffle: bool=True, 
        drop_index: bool=True, 
        columns_to_drop: list=None, 
        columns_to_fillna_with_mean: list=None, 
        columns_to_encode: list=None, 
        columns_dummies: list=None) -> pd.DataFrame:
        
        # drop columns and na
        df.drop(columns_to_drop, axis=1, inplace=True)
        if drop_na:
            df.dropna(inplace=True)
        
        # fillna
        if columns_to_fillna_with_mean is not None:
            for col in columns_to_fillna_with_mean:
                df[col] = df[col].fillna(df[col].mean())
        
        # label encoding
        if columns_to_encode is not None:
            label_encoding = preprocessing.LabelEncoder()
            for col in columns_to_encode:
                df[col] = label_encoding.fit_transform(df[col].astype(str))

        # dummies
        df = pd.get_dummies(df, columns=columns_dummies)

        # TODO: add scaling, normalization
        # Robust Scaler vs standard scaler
        # TODO: Outliers detection
        # TODO: Novelties detection in case dataframe passed is df_test

        # reshuffle
        if reshuffle:
            df = df.sample(frac=1).reset_index(drop=drop_index)
        
        # save data
        file_name = "train_processed" if is_training_data else "test_processed"
        self.save_data(df, file_name)
        print(f"Pre-processed data have been saved as {file_name}.csv.")

        # TODO: Add pre-processing summary 
        # (e.g. columns removed, number of rows removed with NA)
        return df

    def save_predictions(self, 
        df_results: pd.DataFrame, 
        file_name: str='results') -> None:

        self.save_data(df_results, file_name, with_index=False)
        output_message = f'Prediction results have been saved at {self.path}/{file_name}.csv'
        print(f'{output_message}. Ready to be submitted!')