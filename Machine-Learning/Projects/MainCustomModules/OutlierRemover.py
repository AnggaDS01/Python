import pandas as pd
from StatAnalyzer import StatAnalyzer

# Mendefinisikan sebuah kelas dengan nama OutlierRemover
class OutlierRemover(StatAnalyzer):
  # Membuat konstruktor yang menerima dataframe dan objek sa
  def __init__(self, data, target=None):
    # Menyimpan dataframe dan objek sa sebagai atribut
    # self.df = df
    # self.sa = sa
    super().__init__(data, target)
    # Mendapatkan nilai IQR dari setiap kolom menggunakan metode table_diagnose dari objek sa
    self.iqr_df = super().table_diagnose(show_dims=False)

  # Membuat metode untuk menghapus outlier dari dataframe berdasarkan kolom tertentu
  def remove_outliers(self, cols):
    # Membuat salinan dari dataframe
    cleaned_df = self.data.copy(deep=True)
    # Melakukan iterasi pada setiap kolom yang ingin dicari outlier-nya
    for col in cols:
      # Mendapatkan batas bawah dan atas IQR dari kolom tersebut
      lower_limit, upper_limit = self.iqr_df[(self.iqr_df['columns'] == col)][['IQR (< lower)', 'IQR (> upper)']].values.ravel()
      # Memfilter dataframe dengan hanya memilih baris yang memiliki nilai di dalam batas IQR dari kolom tersebut
      cleaned_df = cleaned_df[cleaned_df[col].between(lower_limit, upper_limit)]
    # Mengembalikan dataframe yang sudah dibersihkan dari outlier
    return cleaned_df

  # Membuat metode untuk mendeteksi dan menampilkan outlier dalam dataframe berdasarkan kolom tertentu dan kolom target
  def get_outliers(self, target_col):
    # Filter out the columns that have no upper IQR value
    iqr_df = self.iqr_df[~self.iqr_df['IQR (> upper)'].isna()]

    # Initialize an empty list to store the dataframes with outliers
    original_dfs = []
    outlier_dfs = []
    non_outlier_dfs = []

    # Loop through each column in the iqr_df
    for index_col in range(iqr_df.shape[0]):
      # Get the column name
      col_name = iqr_df.iloc[index_col, 0]
      
      # Skip the target column
      if col_name != target_col:
        # Get the lower and upper limits for outliers based on the IQR
        lower_limit, upper_limit = iqr_df[(iqr_df['columns'] == col_name)][['IQR (< lower)', 'IQR (> upper)']].values.ravel()
        
        # Filter out the rows that are not between the lower and upper limits
        original_df = self.data[[col_name, target_col]].sort_values(by=col_name)
        outlier_df = self.data[~self.data[col_name].between(lower_limit, upper_limit)][[col_name, target_col]].sort_values(by=col_name)
        non_outlier_df = self.data[self.data[col_name].between(lower_limit, upper_limit)][[col_name, target_col]].sort_values(by=col_name)
        # If there are any outliers, call describe method
        if outlier_df.shape[0] != 0:
          outlier_dfs.append(outlier_df.describe(include='all')) # append to the list
          non_outlier_dfs.append(non_outlier_df.describe(include='all'))
          original_dfs.append(original_df.describe(include='all'))

    # Concatenate the list of dataframes with outliers along axis=1
    original_dfs = pd.concat(original_dfs, axis=1)
    outlier_dfs = pd.concat(outlier_dfs, axis=1)
    non_outlier_dfs = pd.concat(non_outlier_dfs, axis=1)
    
    # Return the output dataframe
    return original_dfs, outlier_dfs, non_outlier_dfs