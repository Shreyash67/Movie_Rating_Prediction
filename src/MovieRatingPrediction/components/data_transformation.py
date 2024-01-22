from src.MovieRatingPrediction.logger import logging
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class data_transform:
    def __init__(self, data):
        self.data = data
        self.transformed_data = None

    def data_transform(self):
        self.data.dropna(subset=["Name","Year","Duration","Genre","Rating","Votes","Director"], inplace=True)

        self.data['Duration'] = self.data['Duration'].str.replace('min', '').astype(int)
        self.data['Year'] = self.data['Year'].str.strip('()').astype(int)
        self.data['Votes'] = self.data['Votes'].str.replace(",", "").astype(int)

        self.data.drop(["Name","Director","Genre","Actor 1","Actor 2","Actor 3"], inplace=True, axis=1)

        # Store the transformed data
        self.transformed_data = self.data.copy()

    def get_transformed_data(self):
        if self.transformed_data is not None:
            logging.info("Data is successfully Transform")
            return self.transformed_data
        else:
            logging.info("Data has not been transformed yet. Call data_transform method first.")
            return None
        
# Example usage
data = pd.read_csv(r"D:\Movie_Rating_Prediction\notebooks\data\IMDb Movies India.csv",encoding="windows-1252")
data_transformer = data_transform(data)
data_transformer.data_transform()
transformed_data = data_transformer.get_transformed_data()
print(transformed_data)