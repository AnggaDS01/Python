import pandas as pd
import numpy as np
from fancyimpute import  IterativeImputer
import re

# Define a class to map ordinal and dummy variables to mean target value and impute missing values using IterativeImputer
class DataImputer:
    # Define the constructor method
    def __init__(self):
        pass
        # Initialize an empty dictionary for mapping categories

    # Define a method to map and impute in one step
    def map_and_impute(self, dataframe, ordinal_cols, dummy_cols, target, seed, max_iter, initial_strategy):
    # create an empty dictionary to store the mappings
        my_dict = {}
        # loop through the columns that need to be mapped to numerical values
        for col in dataframe[ordinal_cols + dummy_cols]:
            # get the order of the categories based on the mean of the target variable
            order = dataframe.groupby(col)[target].mean().sort_values().index
            # if the column is ordinal and has missing values, insert nan as the first category
            if col in ordinal_cols and dataframe[col].isna().any():
                order = order.insert(0, np.nan)
            # create a mapping from the categories to numerical values
            my_dict[col] = dict(zip(order, range(len(order))))
            # apply the mapping to the column
            dataframe[col] = dataframe[col].map(my_dict[col])
        # set the random seed for reproducibility
        np.random.seed(seed)
        # create an imputer object with the specified parameters
        imputer = IterativeImputer(max_iter=max_iter, initial_strategy=initial_strategy)
        # fit and transform the dataframe values and convert them back to a dataframe
        imputed_df = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
        # return the imputed dataframe
        self.my_dict = my_dict
        return imputed_df
    
    def create_dummies(self, data, prefix=None, columns=None, drop_first=False):
        # Menggunakan fungsi pd.get_dummies() untuk membuat dummy variabel
        try:
            use_data = data.copy(deep=True)
            dummies = pd.get_dummies(data=use_data, prefix=prefix, columns=columns, drop_first=drop_first)
            # Mengembalikan dataframe dengan dummy variabel
            return dummies
        except KeyError as e:
            error_value = e.args[0]
            columns_diff = re.findall("\[([^\]]+)\]", error_value)[0].split(", ")
            columns_diff = [s.strip("'") for s in columns_diff]
            columns_cleaned = list(set(columns).difference(set(columns_diff)))
            use_data = data.copy(deep=True)
            dummies = pd.get_dummies(data=use_data, prefix=prefix, columns=columns_cleaned, drop_first=drop_first, dtype=int)
            # Mengembalikan dataframe dengan dummy variabel
            return dummies

    def drop_cols(self, dataframe, columns_to_drop):
        # Make a copy of the dataframe
        copy_df = dataframe.copy(deep=True)
        # Drop the specified columns
        copy_df = copy_df.drop(columns=columns_to_drop)
        # Return the modified dataframe
        return copy_df

data_imputer = DataImputer()