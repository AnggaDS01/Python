import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from scipy.stats import pearsonr, chi2_contingency, trim_mean
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV


# define a class for statistical analysis


class StatAnalyzer:
    # define the constructor method that takes a data set and a target variable as arguments
    def __init__(self):
        pass

    # define a method to create a table with various information about a data set
    def table_diagnose(self, data, show_dims=True, trimmed_tol=0.1):
        new_df = data.copy(deep=True)

        # apply different functions to each column of the data set and get the results as a DataFrame
        result = (new_df
                  .agg([lambda x: x.isnull().sum(),  # count the number of missing values
                        lambda x: x.nunique(),  # count the number of unique values
                        lambda x: x.dtype,  # get the data type
                        # get the unique values if less than 10
                        lambda x:  x.unique() if x.nunique() < 10 else np.nan,
                        lambda x: x.isna().mean() * 100,  # calculate the percentage of missing values
                        # calculate the standard deviation if numerical,
                        lambda x: x.skew() if np.issubdtype(x, np.number) else np.nan,
                        lambda x: x.std() if (x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the minimum value if numerical
                        lambda x: x.min() if (x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the first quartile if numerical
                        lambda x: x.quantile(.25) if (
                            x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the mean value if numerical
                        lambda x: x.mean() if (x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the mean value if numerical
                        lambda x: trim_mean(
                            x, proportiontocut=trimmed_tol) if np.issubdtype(x, np.number) else np.nan,
                        # calculate the median value if numerical
                        lambda x: x.quantile(.5) if (
                            x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the third quartile if numerical
                        lambda x: x.quantile(.75) if (
                            x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the maximum value if numerical
                        lambda x: x.max() if (x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the lower bound of interquartile range for each numerical feature
                        lambda x: self.__IQR(
                            x)[0] if (x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the upper bound of interquartile range for each numerical feature
                        lambda x: self.__IQR(
                            x)[1] if (x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        # calculate the mode for each categorical feature
                        lambda x: self.__percentage_outlier(
                            x) if (x.dtype != 'object' and x.dtype != 'category') else np.nan,
                        lambda x: x.mode()[
                      0] if not np.issubdtype(x, np.number) else np.nan,
                  ])
                  ).T.reset_index()  # transpose and reset the index of the DataFrame

        # define a list of column names for clarity
        col_names = [
            'columns',
            'n null',
            'n unique',
            'types',
            'unique vals',
            'percentages null (%)',
            'skewness',
            'std',
            'min',
            'Q1 (25%)',
            'mean',
            'trimmed mean',
            'meadian (Q2)',
            'Q3 (75%)',
            'max',
            'IQR (< lower)',
            'IQR (> upper)',
            'outliers (%)',
            'mode'
        ]

        data_types_list = ['number', 'object', 'datetime',
                           'timedelta', 'category', 'datetimetz']
        data_types_dict = {'rows': len(data), 'columns': len(data.columns)}
        data_types_dict.update({data_type: len(data.select_dtypes(
            include=data_type).columns) for data_type in data_types_list})

        # display a DataFrame with the summary information about the data set
        if show_dims:
            display(pd.DataFrame.from_dict(data_types_dict,
                    orient='index', columns=['Count']).T)

        # rename the columns of the result DataFrame according to the list of column names
        result = result.set_axis(col_names, axis=1)

        # return the result DataFrame with various information about each feature in the data set
        return result

    # define a method to create a DataFrame with various statistical information for each feature in a data set with respect to a target variable
    def feature_relationship_analysis(self, data, target):
        """Perform feature relationship analysis on a data frame.

        This function performs feature relationship analysis on either the numerical or categorical features
        of a data frame, depending on the data type of the target variable. It returns a data frame with the
        source, significance, and p-value of each feature.

        Parameters:
        data (pd.DataFrame): The data frame to perform feature relationship analysis on.
        target (str): The name of the target variable.

        Returns:
        pd.DataFrame: A data frame with the source, significance, and p-value of each feature.

        Examples:
        >>> df = pd.DataFrame({'Island': ['A', 'A', 'B', 'B', 'C', 'C'], 'Height': [10, 12, 14, 16, 18, 20], 'Weight': [20, 22, 24, 26, 28, 30]})
        >>> feature_relationship_analysis(df, 'Island')
                    Significant   p_value
        Height (anova)   15.000  0.003333
        Weight (anova)   15.000  0.003333
        >>> feature_relationship_analysis(df, 'Height')
                        Significant   p_value
        Island (chi2)       0.000     1.000
        Weight (pearson)    1.000     0.000
        """

        # Check the data type of the target variable
        if data[target].dtype == 'object':
            # For categorical target, perform ANOVA and chi-squared analysis
            anova_result = self.__anova(data=data, target=target)
            chi2_result = self.__chi2_stat(data=data, target=target)
            combined_result = pd.concat([anova_result, chi2_result])
        else:
            # For numerical target, perform ANOVA and Pearson correlation analysis
            anova_result = self.__anova(data=data, target=target)
            pearson_result = self.__pearsonr_stat(
                data=data, target=target)
            combined_result = pd.concat([anova_result, pearson_result])

        return combined_result  # Return the result

    # Define a function to perform pairwise Tukey HSD test
    def tukey_hsd_test(self, data, target, alpha=0.05):
        """Perform pairwise Tukey HSD test on a data frame.

        This function performs pairwise Tukey HSD test on the numerical features of a data frame
        with respect to a target column. It returns a data frame with the results of the test.

        Parameters:
        data (pd.DataFrame): The data frame to perform Tukey HSD test on.
        target (str): The name of the target column.
        alpha (float): The significance level for the test.

        Returns:
        pd.DataFrame: A data frame with the results of the test.

        Examples:
        >>> df = pd.DataFrame({'Species': ['A', 'A', 'B', 'B', 'C', 'C'], 'Height': [10, 12, 14, 16, 18, 20], 'Weight': [20, 22, 24, 26, 28, 30]})
        >>> tukey_hsd_test(df, 'Species', alpha=0.05)
                    group1 group2 reject
            Height      A       B   True
                        A       C   True
                        B       C   True
            Weight      A       B   True
                        A       C   True
                        B       C   True
        """

        # Select only the numerical columns from the data frame
        numeric_df = data.select_dtypes(include='number')

        # Create an empty list to store the results of the test
        tukey_results_list = []

        # Loop through each numerical column in the data frame
        for col_num in numeric_df.columns:

            # Perform pairwise Tukey HSD test for each column with respect to the target column
            tukey = pairwise_tukeyhsd(
                endog=data[col_num],  # The response variable
                groups=data[target],  # The group variable
                alpha=alpha  # The significance level
            )

            # Convert the results of the test into a data frame with appropriate columns and index
            tukey_results_list.append(pd.DataFrame(
                # The data from the test results table
                data=tukey._results_table.data[1:],
                # The column names from the test results table
                columns=tukey._results_table.data[0],
                # The index names based on the column name and number of groups
                index=[col_num]*data[target].nunique()
            ))

        # Concatenate all the data frames in the list into one large data frame
        tukey_results_table = pd.concat(tukey_results_list).drop(
            columns=['meandiff', 'p-adj',	'lower', 'upper'])  # Drop some unnecessary columns

        # Return the final data frame with the results of the test
        return tukey_results_table

    def tuning_classifier_models(self, X, y, max_splits=5, test_size=.25, random_state=42, **kwargs):
        """
        Tune and evaluate multiple machine learning models using cross-validation.

        Parameters:
        - X (array-like or pd.DataFrame): Input features.
        - y (array-like or pd.Series): Target labels.
        - model_params (list of dictionaries): List of dictionaries containing model name and corresponding model instance.
        - max_splits (int, optional): Number of cross-validation splits. Default is 5.
        - test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.20.
        - random_state (int, optional): Seed for random number generator. Default is 42.

        Returns:
        - df_scoring (pd.DataFrame): A DataFrame containing accuracy scores of different models for each cross-validation split.
        - models_mean_list (list): A list of strings showing the mean accuracy scores for each model.
        """

        model_params = [
            {'model_name': 'DecisionTreeClassifier',
                'model': DecisionTreeClassifier()},
            {'model_name': 'RandomForestClassifier',
                'model': RandomForestClassifier()},
            {'model_name': 'LogisticRegression', 'model': LogisticRegression()},
            {'model_name': 'SVM', 'model': SVC()},
            {'model_name': 'KNeighborsClassifier',
                'model': KNeighborsClassifier()},
            {'model_name': 'GradientBoostingClassifier',
                'model': GradientBoostingClassifier()},
            {'model_name': 'AdaBoostClassifier', 'model': AdaBoostClassifier()},
            {'model_name': 'GaussianNB', 'model': GaussianNB()},
            {'model_name': 'RidgeClassifier', 'model': RidgeClassifier()},
            {'model_name': 'PassiveAggressiveClassifier',
                'model': PassiveAggressiveClassifier()},
        ]
        rs = ShuffleSplit(n_splits=max_splits, test_size=test_size,
                          random_state=random_state, **kwargs.get('ShuffleSplitParams', {}))
        estimated_model = {}
        get_kind_scoring = kwargs.get(
            'CrossValidateParams', {'scoring': 'accuracy'})

        # Iterate through the provided list of model parameters
        for idx in range(len(model_params)):
            # Perform cross-validation for the current model
            cv_results = cross_validate(
                model_params[idx]['model'], X, y, cv=rs, **get_kind_scoring
            )
            # Store the accuracy scores in the dictionary
            estimated_model[model_params[idx]
                            ['model_name']] = cv_results['test_score']

        # Create a DataFrame to store the accuracy scores for different models
        df_scoring = pd.DataFrame(estimated_model, index=[
                                  f"score {get_kind_scoring['scoring']} {i+1}" for i in range(max_splits)])

        models_mean_list = []

        # Calculate and store the mean accuracy scores for each model
        for col in df_scoring.columns:
            models_mean_list.append(
                f'{col}: {round(df_scoring[col].mean(), 5)}')

        return df_scoring, models_mean_list

    def perform_grid_search(self, X_train, y_train, model, param_grid, num_folds=5, num_jobs=4, **kwargs):
        """
        Perform grid search cross-validation for hyperparameter tuning.

        Parameters:
        - X_train (array-like or pd.DataFrame): Training features.
        - y_train (array-like or pd.Series): Training target labels.
        - model (estimator object): The estimator object with the model to be tuned.
        - param_grid (dict or list of dictionaries): Dictionary or list of dictionaries with parameter names as keys
        and lists of parameter settings to try as values.
        - num_folds (int, optional): Number of cross-validation folds. Default is 5.
        - num_jobs (int, optional): Number of CPU cores to be used for parallel computation. Default is 4.
        - **kwargs: Additional keyword arguments to be passed to GridSearchCV.

        Returns:
        - pd.DataFrame: A DataFrame containing the best score, best parameters, and model information.
        """
        results = []

        # Create a GridSearchCV instance with the given model, parameter grid, and other settings
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=num_folds,
            n_jobs=num_jobs,
            **kwargs
        )

        # Fit the GridSearchCV instance to the training data
        grid_search.fit(X_train, y_train)

        # Store the best score, best parameters, and model information in a list
        results.append({
            'model': str(model),
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        })

        # Create a DataFrame from the list of results
        return pd.DataFrame(results, columns=['model', 'best_score', 'best_params'])

    # define a method to calculate the interquartile range of a series
    def __IQR(self, x):
        # calculate the first and third quartiles
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        # calculate the interquartile range
        iqr = q3 - q1
        # calculate the lower and upper bounds for outliers
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        # return the lower and upper bounds as a tuple
        return lower_bound, upper_bound

    def __percentage_outlier(self, x):
        lower_bound, upper_bound = self.__IQR(x)
        lengt_outlier = np.sum((x < lower_bound) | (x > upper_bound))
        lengt_data = np.sum((x > lower_bound) | (x < upper_bound))
        return lengt_outlier / lengt_data * 100

    def __anova(self, data, target):
        """
        Perform ANOVA (Analysis of Variance) between numerical or categorical features and a target variable.

        This function calculates the F-statistic and p-value for each numerical or categorical feature with
        respect to the given target variable. The F-statistic measures the variance between group means divided
        by the variance within groups, indicating whether there are significant differences in means across groups.
        The p-value helps determine the statistical significance of these differences.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing both the features and the target variable.
        target (str): The name of the target variable column.

        Returns:
        pd.DataFrame: A DataFrame containing the F-statistic and p-value for each feature.

        Note:
        For numerical target variables, ANOVA is performed by comparing the group means of numerical features.
        For categorical target variables, ANOVA is performed by comparing the means of the target variable across
        categories of categorical features.

        Example:
        >>> import pandas as pd
        >>> from sklearn.feature_selection import f_classif
        >>> data = pd.DataFrame({
        ...     'Feature1': [1, 2, 3, 4, 5],
        ...     'Feature2': [10, 20, 30, 40, 50],
        ...     'Category': ['A', 'B', 'A', 'B', 'A'],
        ...     'Target': ['X', 'Y', 'X', 'Y', 'X']
        ... })
        >>> result_df = anova(data, 'Target')
        >>> print(result_df)
                    Significant   p_value
        Feature1 (anova)    0.0  1.000000
        Feature2 (anova)    0.0  1.000000
        Category (anova)    2.0  0.238994
        """

        if data[target].dtype == 'object':  # If the target column is categorical
            numerical_data = data.select_dtypes(
                include='number')  # Select only the numerical columns
            # Calculate the F-statistic and p-value for each numerical column
            F, p = f_classif(numerical_data, data[target])
            # Create a data frame with the results
            result = pd.DataFrame({"Significant": F, "p_value": p})
            result.index = numerical_data.columns + \
                ' (anova)'  # Add the source name to the index
            return result  # Return the result
        else:  # If the target column is numerical
            categorical_data = data.select_dtypes(
                include='object')  # Select only the categorical columns
            # Apply a lambda function to calculate the F-statistic and p-value for each categorical column that has more than one unique value
            result = categorical_data.apply(lambda x: f_classif(
                data[target].to_numpy().reshape(-1, 1), x) if x.nunique() > 1 else 0, axis=0)
            result = result.T  # Transpose the result
            result.columns = ["Significant", "p_value"]  # Add the column names
            # Rename the index with the source name and apply a lambda function to extract the first element of each tuple in the result
            return result.rename(index='{} (anova)'.format).applymap(lambda x: x[0] if type(x) != int else x)

    def __chi2_stat(self, data, target):
        """
        Calculate the chi-squared statistic and p-value for categorical features with respect to a target variable.

        This function calculates the chi-squared statistic and corresponding p-value for each categorical feature
        in relation to the given target variable. The chi-squared test helps determine if there is a significant
        association between categorical variables.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing both the categorical features and the target variable.
        target (str): The name of the target variable column.

        Returns:
        pd.DataFrame: A DataFrame containing the chi-squared statistic and p-value for each categorical feature.

        Example:
        >>> import pandas as pd
        >>> import scipy.stats as stats
        >>> data = pd.DataFrame({
        ...     'Category1': ['A', 'B', 'A', 'B', 'A'],
        ...     'Category2': ['X', 'Y', 'X', 'Y', 'X'],
        ...     'Target': ['Yes', 'No', 'No', 'Yes', 'Yes']
        ... })
        >>> result_df = chi2_stat(data, 'Target')
        >>> print(result_df)
                    Significant   p_value
        Category1 (chi2)         0.0  1.000000
        Category2 (chi2)         0.0  1.000000
        """

        # Select only the categorical columns
        categorical_data = data.select_dtypes(include='object')
        result = categorical_data.apply(lambda x: (  # Apply a lambda function to each column
            # Calculate the chi-square statistic
            chi2_contingency(pd.crosstab(
                x, categorical_data[target])).statistic,
            # Calculate the p-value
            chi2_contingency(pd.crosstab(
                x, categorical_data[target])).pvalue
        ), axis=0).T  # Transpose the result
        result.columns = ["Significant", "p_value"]  # Add the column names
        # Rename the index with the source name
        return result.rename(index='{} (chi2)'.format)

    def __pearsonr_stat(self, data, target):
        """
        Calculate the Pearson correlation coefficient and its p-value for numerical features and a target variable.

        This function calculates the Pearson correlation coefficient and its corresponding p-value for each numerical
        feature with respect to the given target variable. The Pearson correlation measures the linear relationship
        between two variables. The p-value helps determine if the observed correlation is statistically significant.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing both the numerical features and the target variable.
        target (str): The name of the target variable column.

        Returns:
        pd.DataFrame: A DataFrame containing the Pearson correlation coefficient and p-value for each numerical feature.

        Example:
        >>> import pandas as pd
        >>> from scipy import stats
        >>> data = pd.DataFrame({
        ...     'Feature1': [10, 15, 20, 25, 30],
        ...     'Feature2': [5, 10, 15, 20, 25],
        ...     'Target': [50, 60, 45, 55, 65]
        ... })
        >>> result_df = pearsonr_stat(data, 'Target')
        >>> print(result_df)
                    Significant   p_value
        Feature1 (pearson)    0.5  0.282843
        Feature2 (pearson)    0.5  0.282843
        """

        numerical_data = data.select_dtypes(
            include='number')  # Select only the numerical columns
        result = numerical_data.apply(lambda x: pearsonr(
            x, data[target]), axis=0)  # Apply a lambda function to each column
        result = result.T  # Transpose the result
        result.columns = ["Significant", "p_value"]  # Add the column names
        # Rename the index with the source name
        return result.rename(index='{} (pearson)'.format)
