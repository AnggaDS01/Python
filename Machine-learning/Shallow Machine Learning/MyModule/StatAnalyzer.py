import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from scipy.stats import pearsonr, chi2_contingency
import pingouin as pg

# define a class for statistical analysis
class StatAnalyzer:
    # define the constructor method that takes a data set and a target variable as arguments
    def __init__(self, data, target):
        # assign the data set and the target variable as attributes
        self.data = data
        self.target = target
    
    # define a method to calculate the interquartile range of a series
    def IQR(self, x):
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
    
    # define a method to calculate the Pearson correlation coefficient and the p-value for each numerical feature with respect to the target variable
    def pearsonr_stat(self, data):
        # select only the numerical features from the data set
        numerical_data = data.select_dtypes(exclude='object')
        # fill in the missing values with the median
        # numerical_data = numerical_data.fillna(numerical_data.median())
        # create an empty numpy array to store the Pearson correlation results for each numerical feature
        pearsonr_array = np.empty((0,2))
        # iterate over each numerical feature
        for col in numerical_data.columns:
            # calculate the Pearson correlation coefficient and the p-value between the numerical feature and the target variable
            r, p = pearsonr(numerical_data[col], data[self.target])
            # append the Pearson correlation results to the numpy array
            pearsonr_array = np.append(pearsonr_array, [[r, p]], axis=0)
        # get the column names of the numerical features
        columns_name = numerical_data.columns
        # combine the column names and the Pearson correlation results in one DataFrame
        result = pd.DataFrame({"Source": columns_name, "F": pearsonr_array[:,0], "p_value": pearsonr_array[:,1]})
        # return the DataFrame with Pearson correlation results
        return result
    
    # define a method to calculate the eta squared value and the p-value for each categorical feature with respect to the target variable
    def eta_squared(self, data):
        # select only the categorical features from the data set
        categorical_data = data.select_dtypes(include='object')
        # fill in the missing values with the mode
        # categorical_data = categorical_data.fillna(categorical_data.mode())
        # create an empty list to store the ANOVA results for each categorical feature
        result_list = []
        # iterate over each categorical feature
        for col in categorical_data.columns:
            # perform ANOVA for the categorical feature with respect to the target variable
            result = pg.anova(data=data, dv=self.target, between=col)
            # append the ANOVA results to the list
            result_list.append(result)
        # concatenate all the ANOVA results in one DataFrame
        result_df = pd.concat(result_list, ignore_index=True)
        # rename the column name for p-value that is inconsistent
        result_df.rename(columns={'p-unc':'p_value'}, inplace=True)
        # return the DataFrame with ANOVA results with source, F-value, and p-value columns only 
        return result_df[['Source', 'F', 'p_value']]
    
    # define a method to perform ANOVA for each numerical feature with respect to a categorical target variable and get the F-value and p-value 
    def anova(self, data):
        # select only the numerical features from the data set 
        numerical_data = data.select_dtypes(exclude='object')
        # fill in the missing values with median 
        # numerical_data = numerical_data.fillna(numerical_data.median())
 
        # separate the target variable from the data set
        y = data[self.target]
        # convert the target variable to a one-dimensional numpy array
        y = y.to_numpy().ravel()
        # perform ANOVA for each feature with respect to the target variable
        F, p = f_classif(numerical_data, y)
        # get the column names of the numerical features
        columns_name = numerical_data.columns
        # combine the column names, F-values, and p-values in one DataFrame
        result = pd.DataFrame({"Source": columns_name, "F": F, "p_value": p})
        # return the DataFrame with ANOVA results
        return result
    
    # define a method to perform chi-square test for each categorical feature with respect to a categorical target variable and get the chi-square value and p-value
    def chi2_stat(self, data):
        # select only the categorical features from the data set
        categorical_data = data.select_dtypes(include='object')
        # fill in the missing values with the mode
        # categorical_data = categorical_data.fillna(categorical_data.mode())
        # create an empty list to store the chi-square results for each categorical feature
        result_list = []
        # iterate over each categorical feature
        for col in categorical_data.columns:
            # create a contingency table between the categorical feature and the target variable
            contingency_table = pd.crosstab(categorical_data[col], categorical_data[self.target])
            # calculate the chi-square value and p-value from the contingency table
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            # append the chi-square results to the list
            result_list.append((chi2, p))
        # get the column names of the categorical features
        columns_name = categorical_data.columns
        # convert the list of chi-square results to a numpy array
        result_array = np.array(result_list)
        # combine the column names and chi-square results in one DataFrame
        result = pd.DataFrame({"Source": columns_name, "F": result_array[:,0], "p_value": result_array[:,1]})
        # return the DataFrame with chi-square results
        return result

    # define a method to create a table with various information about a data set
    def table_diagnose(self, data=None, show_dims=True):
        if data is None:
            new_df = self.data.copy(deep=True)
        else:
            new_df = data.copy(deep=True)
            
        # apply different functions to each column of the data set and get the results as a DataFrame
        result = (new_df
                .agg([lambda x: x.isnull().sum(), # count the number of missing values
                    lambda x: x.nunique(), # count the number of unique values
                    lambda x: x.dtype, # get the data type
                    lambda x:  x.unique() if x.nunique() < 10 else np.nan, # get the unique values if less than 10
                    lambda x: x.isna().mean() * 100, # calculate the percentage of missing values
                    lambda x: x.std() if x.dtype != 'object' else np.nan, # calculate the standard deviation if numerical
                    lambda x: x.min() if x.dtype != 'object' else np.nan, # calculate the minimum value if numerical
                    lambda x: x.quantile(.25) if x.dtype != 'object' else np.nan, # calculate the first quartile if numerical
                    lambda x: x.mean() if x.dtype != 'object' else np.nan, # calculate the mean value if numerical
                    lambda x: x.quantile(.5) if x.dtype != 'object' else np.nan, # calculate the median value if numerical
                    lambda x: x.quantile(.75) if x.dtype != 'object' else np.nan, # calculate the third quartile if numerical
                    lambda x: x.max() if x.dtype != 'object' else np.nan, # calculate the maximum value if numerical
                    lambda x: self.IQR(x)[0] if x.dtype != 'object' else np.nan, # calculate the lower bound of interquartile range for each numerical feature 
                    lambda x: self.IQR(x)[1] if x.dtype != 'object' else np.nan, # calculate the upper bound of interquartile range for each numerical feature 
                    ])
        ).T.reset_index() # transpose and reset the index of the DataFrame
        
        # define a list of column names for clarity
        col_names = [
            'columns', 
            'n null', 
            'n unique', 
            'types', 
            'unique vals', 
            'percentages null (%)',
            'std',
            'min',
            'Q1',
            'mean',
            'meadian (Q2)',
            'Q3',
            'max',
            'IQR (< lower)', 
            'IQR (> upper)', 
        ]

        # select only the numerical features from the data set
        numerical_data = new_df.select_dtypes(exclude='object')
        # select only the categorical features from the data set
        categorical_data = new_df.select_dtypes(include='object')
        
        # create a tuple with the number of rows and columns in the data set and the number of numerical and categorical features
        t = new_df.shape, numerical_data.shape[1], categorical_data.shape[1]
        new_t = t[0] + t[1:]
        
        # display a DataFrame with the summary information about the data set
        if show_dims:
            display(pd.DataFrame(new_t, index=['Rows', 'Columns', 'Numeric', 'Object'], columns=['Count']).T)
        
        # rename the columns of the result DataFrame according to the list of column names
        result = result.set_axis(col_names, axis=1)
        
        # return the result DataFrame with various information about each feature in the data set
        return result

    # define a method to create a DataFrame with various statistical information for each feature in a data set with respect to a target variable
    def concatinated_df(self, data=None):

        if data is None:
            use_data = self.data
        else:
            use_data = data
        
        # select only the numerical features from the data set 
        numerical_data = use_data.select_dtypes(exclude='object')
        # fill in the missing values with median 
        # numerical_data = numerical_data.fillna(numerical_data.median())
        
        # calculate the skewness of each numerical feature 
        get_skewness = numerical_data.apply(lambda x: x.skew())
        # calculate the lower bound of interquartile range for each numerical feature 
        get_lower_iqr = numerical_data.apply(lambda x: self.IQR(x)[0])
        # calculate the upper bound of interquartile range for each numerical feature 
        get_upper_iqr = numerical_data.apply(lambda x: self.IQR(x)[1])

        try:
            # check if the target variable is numerical 
            if use_data[self.target].dtype != 'object': 
                # calculate the Pearson correlation coefficient and p-value for each numerical feature with respect to the target variable 
                if (use_data.drop(columns=self.target).dtypes != 'object').sum() > 0:
                    get_pearsonr = self.pearsonr_stat(data=use_data)
                else:
                    get_pearsonr = pd.DataFrame([])

                # calculate the eta squared value and p-value for each categorical feature with respect to the target variable 
                if (use_data.drop(columns=self.target).dtypes == 'object').sum() > 0:
                    get_eta2 = self.eta_squared(data=use_data)
                else:
                    get_eta2 = pd.DataFrame([])
                # concatenate the results of Pearson correlation and eta squared in one DataFrame and set source column as index 
                concatinated = pd.concat([get_pearsonr, get_eta2]).set_index('Source')
            # check if target variable is categorical 
            elif use_data[self.target].dtype == 'object':
                # perform ANOVA for each numerical feature with respect to target variable and get F-value and p-value 
                if (use_data.drop(columns=self.target).dtypes != 'object').sum() > 0:
                    get_anova = self.anova(data=use_data)
                else:
                    get_anova = pd.DataFrame([])
                # perform chi-square test for each categorical feature with respect to target variable and get chi-square value and p-value 
                if (use_data.drop(columns=self.target).dtypes == 'object').sum() > 0:
                    get_chi2 = self.chi2_stat(data=use_data)
                else:
                    get_chi2 = pd.DataFrame([])
                # concatenate results of ANOVA and chi-square in one DataFrame and set source column as index 
                concatinated = pd.concat([get_anova, get_chi2]).set_index('Source')
            # reindex concatenated DataFrame according to order of columns in data set 
            concatinated_reindexed = concatinated.reindex(use_data.columns)
            # concatenate reindexed DataFrame with skewness and interquartile range information in one DataFrame 
            result =  pd.concat([concatinated_reindexed, get_skewness, get_lower_iqr, get_upper_iqr], axis=1)
            # rename columns for clarity 
            result = result.rename(columns={0:'skewness', 1:'IQR (< lower)', 2:'IQR (> upper)'})
            # return final DataFrame 
            return result
        except ValueError as e:
            # raise error message if there are still missing values in data set 
            raise ValueError("Periksa kembali apakah data yang anda miliki masih terdapat nilai NaN?")