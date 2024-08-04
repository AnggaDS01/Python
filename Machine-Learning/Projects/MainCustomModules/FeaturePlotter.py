import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency
import seaborn as sns
import tkinter as tk
import pandas as pd

# mendefinisikan Sebuah kelas yang dinamai FeaturePlotter


class FeaturePlotter:
    def __init__(self):
        pass

    def plot_features(self, data, plot_type='scatter', **kwargs):
        cat_features = data.select_dtypes(exclude='number')
        num_features = data.select_dtypes(exclude='object')

        if plot_type == 'scatter':
            num_subplots = len(
                list(combinations_with_replacement(num_features, 2)))
            feature_pairs = combinations_with_replacement(num_features, 2)
        else:
            num_subplots = len(data.columns)

        num_rows = int(np.sqrt(num_subplots))
        num_cols = int(np.ceil(num_subplots / num_rows))
        default_figsize = plt.figure().get_size_inches()
        figsize = (default_figsize[0] * num_rows,
                   default_figsize[1] * num_cols)
        plt.figure(figsize=figsize)

        # Check if the number of feature pairs is too large
        if plot_type == 'scatter':
            self.__plot_scatter(data, feature_pairs,
                                num_cols, num_rows, **kwargs)
        else:
            for i, feature in enumerate(data):
                plt.subplot(num_rows, num_cols, i+1)
                if plot_type == 'boxplot' and np.issubdtype(data[feature], np.number):
                    sns.boxplot(data=num_features, x=feature, **kwargs)
                elif plot_type == 'histogram' and np.issubdtype(data[feature], np.number):
                    sns.histplot(data=num_features, x=feature, **kwargs)
                elif not np.issubdtype(data[feature], np.number):
                    self.__plot_count(feature, cat_features)
        # Adjust the layout and show the figure
        plt.tight_layout()
        plt.show()

    # Definisikan fungsi plot_histograms
    def plot_histograms(self, data, target_col, feature_cols, sample_mean=False, **kwargs):
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
                    random_samples = np.random.choice(
                        species_data, size=(data.shape[0], sample_mean))
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

    def numeric_matrix_corr(self, data, cmap=sns.diverging_palette(
            20, 220, as_cmap=True), method='pearson', **kwargs):
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

    def __plot_scatter(self, data, feature_pairs, num_cols, num_rows, **kwargs):
        for idx, col in enumerate(list(feature_pairs)):
            plt.subplot(num_rows, num_cols, idx+1)
            sns.scatterplot(data=data, x=col[0], y=col[1], **kwargs)

    def __plot_count(self, feature, cat_features, **kwargs):
        ax = sns.countplot(data=cat_features, x=feature, **kwargs)
        ax.xaxis.set_tick_params(rotation=90)
        # Loop over the bars and add text labels
        for p in ax.patches:
            # Get the coordinates of the bar
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + p.get_height() + 0.02
            # Get the count value
            value = int(p.get_height())
            # Add text label with count value
            ax.text(x, y, value, ha="center", va="bottom")
