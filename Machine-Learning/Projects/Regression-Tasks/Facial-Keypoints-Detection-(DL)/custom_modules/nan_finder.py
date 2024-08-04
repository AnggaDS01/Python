# nan_finder.py

import pandas as pd

def find_nan_columns(df, name_columns='Column', max_display=200):
    """
    Find columns with NaN values and display the number or list of NaN-containing columns.

    Args:
        df (pd.DataFrame): DataFrame to search for NaN values.
        name_columns (str, optional): Name for the index column in the output. Default is 'Column'.
        max_display (int, optional): Maximum number of NaN columns to display as a list. If exceeded, only the count is shown. Default is 200.

    Returns:
        dict: Dictionary where keys are column names and values are either a list of columns with NaN values or a string indicating the count of NaN values.

    Example:
        >>> import pandas as pd
        >>> from nan_finder import find_nan_columns

        >>> data = {
        ...     'A': [1, 2, None],
        ...     'B': [None, 2, 3],
        ...     'C': [1, 2, 3]
        ... }
        >>> df = pd.DataFrame(data)
        >>> result = find_nan_columns(df)
        >>> print(result)
        Column:
        A:   B
        B:   A
        """
    
    # Make a deep copy of the DataFrame
    df_copy = df.copy(deep=True)
    
    # Transpose the DataFrame and reset the index
    df_copy_transformed = df_copy.T.reset_index().rename(columns={'index': name_columns})
    
    # Initialize a dictionary to hold columns with NaN values
    nan_columns = {}
    
    # Iterate over the transposed DataFrame rows
    for index, row in df_copy_transformed.iterrows():
        # Find columns with NaN values
        nan_cols = row.index[row.isna()].tolist()
        
        # Store the NaN columns or the count if it exceeds max_display
        if len(nan_cols) > max_display:
            nan_columns[row[name_columns]] = f"Jumlah NaN: {len(nan_cols)}"
        else:
            nan_columns[row[name_columns]] = nan_cols

    # Display the columns with NaN values
    for key, value in nan_columns.items():
        if isinstance(value, list):
            value_str = ' '.join(map(str, value))
        else:
            value_str = value
        print(f"{key}:\t{value_str}")
    
    return nan_columns
