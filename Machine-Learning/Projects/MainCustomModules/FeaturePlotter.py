import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency
import seaborn as sns
import tkinter as tk
import pandas as pd

# mendefinisikan Sebuah kelas yang dinamai FeaturePlotter


class FeaturePlotter:
    def __init__(self):
        pass

    def plots_count_features(self, data, hue=None, figsize_per_image=(5, 7), left_idx=0, right_idx=None, show_grid=False, stat='count', **countplot_kwargs):
        categorical_data=data.select_dtypes(exclude='number')
        if right_idx is None:
            right_idx = categorical_data.shape[1]
        data_columns_categorical_features=categorical_data.iloc[:, left_idx:right_idx].columns

        # Tentukan jumlah subplot
        num_subplots = len(data_columns_categorical_features)

        # Tentukan ukuran grid subplot
        num_rows = int(np.sqrt(num_subplots))
        num_cols = int(np.ceil(num_subplots / num_rows))
        # Tentukan ukuran figure        
        figsize = (num_cols * figsize_per_image[0], num_rows * figsize_per_image[1])

        # Plot scatterplot untuk setiap pasangan fitur
        plt.figure(figsize=figsize)

        for idx_subplot, categorical_column_data in enumerate(data_columns_categorical_features, 1):
            ax = plt.subplot(num_rows, num_cols, idx_subplot)
            sns.countplot(data=data, x=categorical_column_data, hue=hue, legend=(idx_subplot==1), ax=ax, stat=stat, **countplot_kwargs)
            if idx_subplot == 1 and hue:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()

            for p in ax.patches:
                if stat == 'count':
                    n_value = f'{p.get_height():.0f}'
                else:
                    n_value = f'{p.get_height():.2f}'
                x = p.get_x() + p.get_width() / 2
                y = p.get_y() + p.get_height()
                ax.text(x, y, n_value, ha="center", va="bottom")
            # Tambahkan padding di bagian atas axes
            y_max = max([p.get_height() for p in ax.patches])
            ax.set_ylim(0, y_max * 1.1)

            plt.xticks(rotation=90)
            plt.grid(show_grid)
        plt.tight_layout()

        if hue:
            plt.figure(figsize=(2,2))
            plt.legend(handles, labels, loc='upper center', title=hue)
            plt.axis('off')  # Menyembunyikan axis
        plt.show()

    def plots_scatter_features(self, data, hue=None, figsize_per_image=(6, 4), figsize_legend=(2,2), left_idx=0, right_idx=None, **scatter_kwargs):
        numerical_data=data.select_dtypes(exclude='object')
        
        if right_idx is None:
            right_idx = numerical_data.shape[1]

        # Ambil 3 kolom pertama dari fitur numerik
        data_columns_numerical_features=numerical_data.iloc[:, left_idx:right_idx].columns

        # Buat pasangan fitur unik
        feature_pairs = list(combinations(data_columns_numerical_features, 2))

        # Tentukan jumlah subplot
        num_subplots = len(feature_pairs)

        # Tentukan ukuran grid subplot
        num_rows = int(np.sqrt(num_subplots))
        num_cols = int(np.ceil(num_subplots / num_rows))

        # Tentukan ukuran figure        
        figsize = (num_cols * figsize_per_image[0], num_rows * figsize_per_image[1])

        # Plot scatterplot untuk setiap pasangan fitur
        plt.figure(figsize=figsize)
        for idx_subplot, (col1, col2) in enumerate(feature_pairs, 1):
            ax = plt.subplot(num_rows, num_cols, idx_subplot)
            
            sns.scatterplot(data=data, x=col1, y=col2, hue=hue, ax=ax, legend=(idx_subplot==1), **scatter_kwargs)
            if idx_subplot == 1 and hue:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()

            plt.xticks(rotation=25)

        # Menyediakan ruang untuk legend di sebelah kanan
        plt.tight_layout()

        # Buat figure terpisah untuk legend
        if hue:
            plt.figure(figsize=figsize_legend)
            plt.legend(handles, labels, loc='upper center', title=hue)
            plt.axis('off')  # Menyembunyikan axis

        # Tampilkan figure
        plt.show()

    def plots_histograms_features(self, data, hue=None, figsize_per_image=(5, 4), left_idx=0, right_idx=None, show_grid=False, **histplot_kwargs):
        numerical_data=data.select_dtypes(exclude='object')
        
        if right_idx is None:
            right_idx = numerical_data.shape[1]

        data_columns_numerical_features=numerical_data.iloc[:, left_idx:right_idx].columns

        # Tentukan jumlah subplot
        num_subplots = len(data_columns_numerical_features)

        # Tentukan ukuran grid subplot
        num_rows = int(np.sqrt(num_subplots))
        num_cols = int(np.ceil(num_subplots / num_rows))
        # Tentukan ukuran figure        
        figsize = (num_cols * figsize_per_image[0], num_rows * figsize_per_image[1])

        # Plot scatterplot untuk setiap pasangan fitur
        plt.figure(figsize=figsize)

        for idx_subplot, numerical_column_data in enumerate(data_columns_numerical_features, 1):
            plt.subplot(num_rows, num_cols, idx_subplot)
            ax=sns.histplot(data=data, x=numerical_column_data, hue=hue,  legend=(idx_subplot==1),  **histplot_kwargs)
            if idx_subplot == 1 and hue:
                legend = ax.get_legend()
                handles = legend.legend_handles
                labels = [t.get_text() for t in legend.texts]
                ax.legend_.remove()
            plt.xticks(rotation=90)
            plt.grid(show_grid)

        plt.tight_layout()

        if hue:
            plt.figure(figsize=(2,2))
            plt.legend(handles, labels, loc='upper center', title=hue)
            plt.axis('off')  # Menyembunyikan axis
        plt.show()

    def plots_box_features(self, data, hue=None, figsize_per_image=(5, 4), left_idx=0, right_idx=None, show_grid=False, **boxplot_kwargs):
        numerical_data=data.select_dtypes(exclude='object')
        
        if right_idx is None:
            right_idx = numerical_data.shape[1]

        data_columns_numerical_features=numerical_data.iloc[:, left_idx:right_idx].columns

        # Tentukan jumlah subplot
        num_subplots = len(data_columns_numerical_features)

        # Tentukan ukuran grid subplot
        num_rows = int(np.sqrt(num_subplots))
        num_cols = int(np.ceil(num_subplots / num_rows))
        # Tentukan ukuran figure        
        figsize = (num_cols * figsize_per_image[0], num_rows * figsize_per_image[1])

        # Plot scatterplot untuk setiap pasangan fitur
        plt.figure(figsize=figsize)

        for idx_subplot, numerical_column_data in enumerate(data_columns_numerical_features, 1):
            plt.subplot(num_rows, num_cols, idx_subplot)
            ax=sns.boxplot(data=data, x=numerical_column_data, hue=hue,  legend=(idx_subplot==1),  **boxplot_kwargs)
            if idx_subplot == 1 and hue:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()
            plt.xticks(rotation=90)
            plt.grid(show_grid)

        plt.tight_layout()

        if hue:
            plt.figure(figsize=(2,2))
            plt.legend(handles, labels, loc='upper center', title=hue)
            plt.axis('off')  # Menyembunyikan axis
        plt.show()

    # Definisikan fungsi plots_histogram
    def plots_histogram_with_sample_mean(self, data, target_col, feature_cols, sample_mean=False, **kwargs):
        # Loop untuk setiap kolom target
        for target in feature_cols:
            # Buat dataframe kosong untuk menyimpan hasil sampling acak
            new_df = pd.DataFrame()
            # Ubah bentuk dataframe df menjadi pivot_df dengan kolom 'Species' sebagai kolom baru dan nilai target sebagai nilai dalam setiap sel
            pivot_df = data.pivot(columns=target_col, values=target)
            # Loop untuk setiap spesies yang ada di kolom pivot_df
            if sample_mean:
                for species in pivot_df.columns:
                    # Ambil data spesies yang tidak kosong dan simpan dalam variabel species_data
                    species_data = pivot_df[species].dropna()
                    # Ambil sampel acak sebanyak 10 kali dari species_data, dengan jumlah sampel sama dengan jumlah baris di df, dan simpan dalam variabel random_samples
                    random_samples = np.random.choice(species_data, size=(data.shape[0], sample_mean))
                    # Hitung rata-rata dari setiap sampel acak dan simpan dalam variabel random_means
                    random_means = np.mean(random_samples, axis=1)
                    # Tambahkan random_means sebagai kolom baru di new_df dengan nama spesies
                    new_df[species] = random_means
                # Buat histogram dari new_df dengan parameter-parameter tertentu
                new_df.plot(kind='hist', title=target, **kwargs)
            else:
                pivot_df.plot(kind='hist', title=target, **kwargs)

            # Tampilkan gambar yang telah dibuat
            plt.show()

    def cramers_v_matrix(self, data, cmap=sns.diverging_palette(
            20, 220, as_cmap=True), **kwargs):
        """Create a Cramér's V matrix for a data frame.

        This function creates a Cramér's V matrix for a data frame with nominal variables.
        It returns a data frame with the Cramér's V values as the correlation matrix.

        Parameters:
        df (pd.DataFrame): The data frame with nominal variables.

        Returns:
        pd.DataFrame: The data frame with the Cramér's V values as the correlation matrix.

        Examples:
        >>> df = pd.DataFrame({'Island': ['A', 'A', 'B', 'B', 'C', 'C'], 'Gender': ['M', 'F', 'M', 'F', 'M', 'F'], 'Color': ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Pink']})
        >>> cramers_v_matrix(df)
                    Island    Gender     Color
        Island    1.000000  0.000000  0.577350
        Gender    0.000000  1.000000  0.000000
        Color     0.577350  0.000000  1.000000
        """

        # Select only the categorical columns from the data frame
        cat_df = data.select_dtypes(exclude='number')
        # Get a boolean Series with True for columns that have more than one unique value
        mask = cat_df.nunique() > 1
        # Select only those columns from the DataFrame and return a new DataFrame
        df_masked = cat_df.loc[:, mask]

        # Initialize an empty matrix with the same shape as the data frame
        corr_matrix = np.zeros((df_masked.shape[1], df_masked.shape[1]))
        # Loop through the rows and columns of the data frame
        for i in range(df_masked.shape[1]):
            for j in range(df_masked.shape[1]):
                # Get the values of the ith and jth columns as pandas series
                x = df_masked.iloc[:, i]
                y = df_masked.iloc[:, j]
                # Calculate the contingency table, chi-square, and degrees of freedom
                cont_table = pd.crosstab(x, y)
                chi2, _, _, _ = chi2_contingency(cont_table)
                n = cont_table.sum().sum()
                r, k = cont_table.shape
                # Calculate the Cramér's V value and assign it to the matrix element
                v = np.sqrt((chi2 / n) / (min(k-1, r-1)))
                corr_matrix[i, j] = v
        # Convert the matrix to a data frame with the same column names as the input
        corr_df = pd.DataFrame(
            corr_matrix, index=df_masked.columns, columns=df_masked.columns)

        # Membuat heatmap dari matrix correlation
        sns.heatmap(corr_df, annot=True, cmap=cmap, **kwargs)
        # Menambahkan judul dan label sumbu
        plt.title('Cramers v matrix')
        # Menampilkan plot
        plt.show()

    def numeric_matrix_corr(self, data, cmap=sns.diverging_palette(20, 220, as_cmap=True), method='pearson', **kwargs):
        num_df_corr = data.corr(method=method, numeric_only=True)
        plt.title(f'{method.title()} Matrix Correlation')
        sns.heatmap(num_df_corr, annot=True, cmap=cmap, **kwargs)
        plt.show()

    def plot_confusion_matrix_report(
        self,
        y_true,
        y_pred,
        class_labels="auto",
        figsize=(16, 8),
        y_ticks_rot=0,
        x_ticks_rot=0,
        fs_title=20,
        fs_label=14,
        pallete=plt.cm.magma,
        show_classification_report=True,
        **kwargs
    ):
        """
        Plot a confusion matrix along with optional classification report.

        Parameters:
        - matrix (array-like): The confusion matrix to be visualized.
        - class_labels (array-like or "auto", optional): Labels for the classes. If "auto", numeric labels will be used.
        - figsize (tuple, optional): Figure size. Default is (16, 8).
        - y_ticks_rot (float, optional): Rotation angle for y-axis tick labels. Default is 0.
        - x_ticks_rot (float, optional): Rotation angle for x-axis tick labels. Default is 0.
        - fs_title (int, optional): Font size for the plot title. Default is 20.
        - fs_label (int, optional): Font size for the axis labels. Default is 14.
        - palette (colormap, optional): Colormap to be used. Default is plt.cm.magma.
        - show_classification_report (bool, optional): Whether to display the classification report. Default is True.
        - **kwargs: Additional keyword arguments to be passed to the seaborn.heatmap() function.

        Returns:
        - None
        """
        matrix = confusion_matrix(y_true, y_pred)
        # Calculate the percentages and formatted matrix values
        norm = matrix.sum(axis=1, keepdims=True)
        percentages = ((matrix / norm) * 100).ravel()
        matrices = matrix.ravel()
        cm = np.array([f'{val}\n{percentage:.5f}%' for percentage, val in zip(
            percentages, matrices)]).reshape(matrix.shape)

        # If requested, print the classification report
        if show_classification_report:
            print(classification_report(
                y_true, y_pred, target_names=class_labels))

        # Create the heatmap plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            matrix,
            annot=cm,
            cmap=pallete,
            fmt='s',
            xticklabels=class_labels,
            yticklabels=class_labels,
            **kwargs
        )
        plt.xticks(rotation=x_ticks_rot)
        plt.yticks(rotation=y_ticks_rot)
        plt.ylabel('True Label', fontsize=fs_label)
        plt.xlabel('Predicted Label', fontsize=fs_label)
        plt.title('confusion matrix', fontsize=fs_title)
        plt.show()
