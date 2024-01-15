import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TabularDataset(Dataset):
    def __init__(self, data):
        self.original_data = data.copy()
        self.data = data
        self.original_feature_names = data.columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        numerical_columns = self.data.select_dtypes(include=['int64', 'float64']).columns

        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_columns),
            ('cat', categorical_pipeline, categorical_columns)
        ])
        self.fit_preprocessor()

    def fit_preprocessor(self):
        self.preprocessor.fit(self.original_data)
        self.feature_names = self.preprocessor.get_feature_names_out()
        self.data = self.preprocessor.transform(self.original_data)
        
        self.global_min = self.data.min(axis=0)
        self.global_max = self.data.max(axis=0)
        
    def get_numerical_columns(self):
        return self.original_data.select_dtypes(include=['int64', 'float64']).columns
    
    def get_categorical_columns(self):
        return self.original_data.select_dtypes(include=['object', 'category']).columns
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = torch.tensor(self.data[idx], dtype=torch.float32)
        return data_point

def normalize_data(data):
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

    data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].min()) / (data[numerical_columns].max() - data[numerical_columns].min())

    return data

def load_data(file_path, batch_size=32, shuffle=True):
    data = pd.read_csv(file_path)
    data = normalize_data(data)
    dataset = TabularDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
