from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, LabelBinarizer
import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.utils import resample
import re

# Define a class to map ordinal and dummy variables to mean target value and impute missing values using IterativeImputer


class DataImputer:
    # Define the constructor method
    def __init__(self):
        pass

    # Define a method to map and impute in one step
    def map_and_impute(self, data, ordinal_cols, dummy_cols, target, seed, max_iter, initial_strategy):
        # create an empty dictionary to store the mappings
        my_dict = {}
        # loop through the columns that need to be mapped to numerical values
        for col in data[ordinal_cols + dummy_cols]:
            # get the order of the categories based on the mean of the target variable
            order = data.groupby(col)[target].mean().sort_values().index
            # if the column is ordinal and has missing values, insert nan as the first category
            if col in ordinal_cols and data[col].isna().any():
                order = order.insert(0, np.nan)
            # create a mapping from the categories to numerical values
            my_dict[col] = dict(zip(order, range(len(order))))
            # apply the mapping to the column
            data[col] = data[col].map(my_dict[col])
        # set the random seed for reproducibility
        np.random.seed(seed)
        # create an imputer object with the specified parameters
        imputer = IterativeImputer(
            max_iter=max_iter, initial_strategy=initial_strategy)
        # fit and transform the dataframe values and convert them back to a dataframe
        imputed_df = pd.DataFrame(imputer.fit_transform(
            data), columns=data.columns)
        # return the imputed dataframe
        self.my_dict = my_dict
        return imputed_df

    def create_dummies(self, data, prefix=None, columns=None, drop_first=False):
        # Menggunakan fungsi pd.get_dummies() untuk membuat dummy variabel
        try:
            use_data = data.copy(deep=True)
            dummies = pd.get_dummies(
                data=use_data, prefix=prefix, columns=columns, drop_first=drop_first)
            # Mengembalikan dataframe dengan dummy variabel
            return dummies
        except KeyError as e:
            error_value = e.args[0]
            columns_diff = re.findall(
                "\[([^\]]+)\]", error_value)[0].split(", ")
            columns_diff = [s.strip("'") for s in columns_diff]
            columns_cleaned = list(set(columns).difference(set(columns_diff)))
            use_data = data.copy(deep=True)
            dummies = pd.get_dummies(
                data=use_data, prefix=prefix, columns=columns_cleaned, drop_first=drop_first, dtype=int)
            # Mengembalikan dataframe dengan dummy variabel
            return dummies

    # Definisikan fungsi imbalance_data untuk menangani data yang tidak seimbang
    def imbalance_data(self, data, target_column, upsample=True, **kwargs):
        """
        Fungsi ini menerima sebuah dataframe yang memiliki data tidak seimbang pada kolom target,
        dan mengembalikan sebuah list dataframe yang sudah diresampling agar memiliki jumlah sampel yang sama untuk setiap kelas target.
        Parameter:
        - dataframe: dataframe yang berisi data yang ingin diresampling.
        - target_column: nama kolom yang menjadi kelas target.
        - upsample: boolean yang menentukan apakah menggunakan upsampling atau downsampling. Default adalah True, artinya menggunakan upsampling.
        - kwargs: argumen tambahan untuk fungsi resample dari library sklearn.utils.
        Output:
        - resampled_dataframes: list dataframe yang sudah diresampling.
        """
        # Buat list kosong untuk menyimpan hasil resampling
        resampled_dataframes = []
        # Cari jumlah sampel yang diinginkan untuk setiap kelas target
        if upsample:
            # Jika menggunakan upsampling, maka jumlah sampel adalah jumlah maksimum dari semua kelas target
            desired_sample_size = data[target_column].value_counts().max()
        else:
            # Jika menggunakan downsampling, maka jumlah sampel adalah jumlah minimum dari semua kelas target
            desired_sample_size = data[target_column].value_counts().min()

        # Loop untuk setiap nilai dan jumlah dari kelas target
        for value, count in data[target_column].value_counts().to_dict().items():
            # Buat salinan dari dataframe
            new_dataframe = data.copy(deep=True)
            # Cek apakah jumlah kelas target tidak sama dengan jumlah sampel yang diinginkan
            if count != desired_sample_size:
                # Jika tidak sama, maka lakukan resampling dengan fungsi resample
                new_dataframe = resample(
                    # Data yang ingin diresample, yaitu new_dataframe yang sudah difilter berdasarkan nilai kelas target
                    new_dataframe[new_dataframe[target_column] == value],
                    # Jumlah sampel yang diinginkan, yaitu desired_sample_size
                    n_samples=desired_sample_size,
                    # Argumen tambahan yang diberikan oleh kwargs
                    **kwargs
                )
            else:
                # Jika sama, maka tidak perlu resampling, cukup filter new_dataframe berdasarkan nilai kelas target
                new_dataframe = new_dataframe[new_dataframe[target_column] == value]

            # Tambahkan hasil resampling atau filtering ke list resampled_dataframes
            resampled_dataframes.append(new_dataframe)

        # Kembalikan list resampled_dataframes sebagai output fungsi
        return pd.concat(resampled_dataframes).reset_index(drop=True)

    def preprocess_data(self, data, target_column, kind_norm_num='minmax', **kwargs):
        """
        Preprocesses the input data by performing various transformations on both categorical and numerical features.

        Parameters:
            data (DataFrame): The input data to be preprocessed.
            target_column (str): The name of the target column.
            kind_norm_num (str, optional): The type of numerical normalization to be applied. 
                                        Options: 'minmax' (default) or 'standard'.
            **kwargs: Additional keyword arguments for specific preprocessing steps.
                    'Ohe': Dictionary with keyword arguments for pandas get_dummies() function.
                    'MinMax': Dictionary with keyword arguments for MinMaxScaler().
                    'Standard': Dictionary with keyword arguments for StandardScaler().

        Returns:
            DataFrame or tuple: Returns a preprocessed DataFrame. If the target column is categorical, returns a tuple 
                                containing the preprocessed DataFrame and a LabelEncoder object for the target labels.
        """
        std_scaler = StandardScaler()  # StandardScaler for z-score normalization
        mm_scaler = MinMaxScaler()      # MinMaxScaler for min-max normalization
        lb = LabelBinarizer()           # LabelBinarizer for binary categorical features
        enc = LabelEncoder()            # LabelEncoder for categorical target column

        # Create a copy of the input data
        df_preprocessed = data.copy(deep=True)
        # Separate features from target column
        X = df_preprocessed.drop(columns=target_column)

        categorical_columns = X.select_dtypes(
            include=['object', 'category']).columns
        numerical_columns = X.select_dtypes(include='number').columns

        for col in categorical_columns:
            if df_preprocessed[col].nunique() == 2:  # Binary categorical features
                df_preprocessed[col] = lb.fit_transform(
                    df_preprocessed[col]) / 1.0
            # Categorical features with more than two categories
            elif df_preprocessed[col].nunique() > 2:
                df_preprocessed = pd.get_dummies(data=df_preprocessed, columns=[
                                                 col], **kwargs.get('Ohe', {'dtype': float}))

        if kind_norm_num == 'minmax':
            df_preprocessed[numerical_columns] = mm_scaler.fit_transform(
                df_preprocessed[numerical_columns], **kwargs.get('MinMax', {}))
        elif kind_norm_num == 'standar':
            df_preprocessed[numerical_columns] = std_scaler.fit_transform(
                df_preprocessed[numerical_columns], **kwargs.get('Standard', {}))

        # Categorical target column
        if not np.issubdtype(df_preprocessed[target_column].dtype, np.number):
            df_preprocessed[target_column] = enc.fit_transform(
                df_preprocessed[target_column])
            return df_preprocessed, enc  # Return preprocessed data and LabelEncoder object
        else:
            return df_preprocessed  # Return only preprocessed data
