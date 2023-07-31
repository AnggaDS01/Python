import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement
import tkinter as tk

# mendefinisikan Sebuah kelas yang dinamai FeaturePlotter
class FeaturePlotter:
    def __init__(self, data, feature_columns):
        self.data = data
        self.feature_columns = feature_columns

    def plot_features(self, data=None, plot_type='scatter', **kwargs):
        if data is None:
            cat_features = self.data.select_dtypes(include='object')
            num_features = self.data.select_dtypes(exclude='object')
        else:
            cat_features = data.select_dtypes(include='object')
            num_features = data.select_dtypes(exclude='object')
        # Get all possible pairs of features
        feature_pairs = list(combinations_with_replacement(self.feature_columns, 2))

        # Determine the number of subplots based on the plot type
        if plot_type == 'scatter':
            num_subplots = len(feature_pairs)
        else:
            num_subplots = len(self.feature_columns)

        # Calculate the number of rows and columns for the subplots
        num_cols = int(np.sqrt(num_subplots))
        num_rows = np.math.ceil(num_subplots / num_cols)

        # Get the default figure size
        default_figsize = plt.figure().get_size_inches()
        # Adjust the figure size according to the number of subplots
        figsize = (default_figsize[0] * num_cols, default_figsize[1] * num_rows)
        # Create the figure and axes for the subplots
        plt.figure(figsize=figsize)

        # Check if the number of feature pairs is too large
        if len(feature_pairs) > 60 and plot_type == 'scatter':
            # Create a dialog box to ask for confirmation
            answer = self.ask_confirmation()
            # If the answer is yes, proceed with plotting
            if answer == 'yes':
                self.plot_scatter(feature_pairs, num_rows, num_cols, **kwargs)
            # If the answer is no, return without plotting
            else:
                return None
        # If the number of feature pairs is not too large, proceed with plotting
        else:
            if plot_type == 'scatter':
                self.plot_scatter(feature_pairs, num_rows, num_cols, **kwargs)
            else:
                for i, feature in enumerate(self.feature_columns):
                    plt.subplot(num_rows, num_cols, i+1)
                    if plot_type == 'boxplot' and self.data[feature].dtype != 'object':
                        self.plot_box(feature, num_features, **kwargs)
                    elif plot_type == 'histogram' and self.data[feature].dtype != 'object':
                        self.plot_hist(feature, num_features, **kwargs)
                    elif self.data[feature].dtype == 'object':
                        self.plot_count(feature, cat_features)


        # Adjust the layout and show the figure
        plt.tight_layout()
        plt.show()

    def ask_confirmation(self):
        # Create a root window for the dialog box
        root = tk.Tk()
        # Hide the root window from view
        root.withdraw()
        # Create a message to display on the dialog box
        message = f'The number of feature pairs is too large . This may take a long time to plot. Do you want to continue?'
        # Create a title for the dialog box
        title = "Confirmation"
        # Use tkinter.messagebox to create a yes/no dialog box
        answer = tk.messagebox.askquestion(title, message)
        # Return the answer as 'yes' or 'no'
        return answer

    def plot_scatter(self, feature_pairs, num_rows, num_cols, **kwargs):
        # Loop over the subplots and plot the features as scatter plots
        for i, pair in enumerate(feature_pairs):
            plt.subplot(num_rows, num_cols, i+1)
            sns.scatterplot(data=self.data, x=pair[0], y=pair[1], **kwargs)
    
    def plot_box(self, feature, num_features, **kwargs):
        sns.boxplot(data=num_features, x=feature, **kwargs)
    
    def plot_hist(self, feature, num_features, **kwargs):
        sns.histplot(data=num_features, x=feature, **kwargs)

    def plot_count(self, feature, cat_features, **kwargs):
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