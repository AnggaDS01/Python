import numpy as np
import matplotlib.pyplot as plt
# source https://towardsdev.com/how-to-identify-missingness-types-with-missingno-61cfe0449ad9
import missingno as msno

from StatAnalyzer import StatAnalyzer


# Defining a class named MissingDataPlotter that inherits from DataDiagnoser
class MissingDataPlotter(StatAnalyzer):
    # Defining a constructor method that takes a data argument and calls the parent constructor method
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Defining a method that takes a plot_type and **kwargs as arguments
    def missing_plot(self, data, plot_type, **kwargs):
        # Fungsi ini digunakan untuk membuat plot data yang hilang dengan library missingno
        # plot_type: jenis plot yang ingin dibuat ('matrix', 'heatmap', 'bar', atau 'dendrogram')
        # **kwargs: parameter tambahan yang dapat dikirimkan ke fungsi plot missingno
        if plot_type == 'matrix':
            # Membuat matriks data yang hilang
            msno.matrix(data, **kwargs)
        elif plot_type == 'heatmap':
            # Membuat heatmap data yang hilang
            msno.heatmap(data, **kwargs)
        elif plot_type == 'bar':
            # Membuat barplot data yang hilang
            msno.bar(data, **kwargs)
        elif plot_type == 'dendrogram':
            # Membuat dendrogram data yang hilang
            msno.dendrogram(data, **kwargs)
        else:
            # Menampilkan pesan error jika plot_type tidak valid
            print("Invalid plot type. Please choose one of the following: 'matrix', 'heatmap', 'bar', or 'dendrogram'.")
        plt.show()  # Menampilkan plot

    # Defining a method that takes no arguments
    def plot_missing_data(self, data, is_null=True):
        # Membuat salinan dari dataframe data
        dummy_df_copy = data.copy(deep=True)
        # Mengubah nama kolom menjadi angka urut
        dummy_df = dummy_df_copy.rename(
            columns={col: i+1 for i, col in enumerate(data.columns)})

        # Membuat dataframe yang berisi nama kolom dan jumlah data yang hilang dengan menggunakan metode table_diagnose dari kelas induk
        get_columns_n_null = super().table_diagnose(
            data=dummy_df_copy, show_dims=False).loc[:, ['columns', 'n null']]

        # Membuat dataframe yang berisi nama kolom yang tidak memiliki data yang hilang
        get_columns_not_null = dummy_df.columns if is_null else get_columns_n_null[
            get_columns_n_null['n null'] == 0]['columns']

        # Mendapatkan indeks kolom yang tidak memiliki data yang hilang
        get_index_cols_null = dummy_df.columns + \
            1 if is_null else get_columns_not_null.index + 1
        # Mendapatkan nama kolom yang tidak memiliki data yang hilang
        get_index_cols_values = dummy_df_copy.columns if is_null else get_columns_not_null.values
        # Mendapatkan jumlah kolom yang tidak memiliki data yang hilang
        cols_init = len(get_index_cols_null)

        # Menghitung jumlah baris dan kolom untuk subplot
        cols = int(np.sqrt(cols_init))
        rows = np.math.ceil(cols_init / cols)

        # Mendapatkan ukuran default dari figure
        figsize = plt.figure().get_size_inches()
        # Mengatur ukuran figure sesuai dengan jumlah subplot
        figsize = (figsize[0] * cols * 2.5, figsize[1] * rows*1.5)
        # Membuat figure dan axes untuk subplot
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        # Menggunakan satu for loop untuk membuat subplot
        for index, value in enumerate(dummy_df.columns if is_null else get_index_cols_null):
            # Mendapatkan koordinat baris dan kolom dari indeks
            i, j = divmod(index, cols)
            # Membuat subplot pada posisi i,j dengan menggunakan library missingno
            msno.matrix(dummy_df.sort_values(
                value), ax=axes[i, j], sparkline=False, labels=True, label_rotation=90, fontsize=8)
            # Menambahkan x label pada subplot dengan nama kolom asli
            axes[i, j].set_xlabel(get_index_cols_values[index], fontsize=12)
        # Menyesuaikan jarak antara subplot
        plt.tight_layout()
        # Menampilkan plot
        plt.show()
