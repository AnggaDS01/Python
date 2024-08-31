from utils.postgre_database_connection import connect_to_local_database
from tqdm import tqdm

import pandas as pd
import psycopg2
import glob
import os
import re

get_all_files_path_product = glob.glob('*/assets/datasets/*.csv')

CREATE_TABLE_QUERY = """
--sql
CREATE TABLE amazon_products.{} ()
;
"""

ADD_COLUMN_QUERY = """
--sql
ALTER TABLE amazon_products.{}
    ADD COLUMN {}
;
"""

INSERT_VALUES_QUERY = """
--sql
INSERT INTO amazon_products.{} ({})
VALUES ({})
;
"""

count_data = 0

if __name__ == "__main__":
    try: 
        connection = connect_to_local_database()
        cursor = connection.cursor()
        print("Successfully connected to the database.")
        print("=".center(100, "="))
        
        for file_path in get_all_files_path_product:
            table_name = file_path.split(os.path.sep)[-1][:-4]
            table_name = re.sub(r"-|\s+", "_", table_name).lower()
            
            if table_name != 'amazon_products':
                df = pd.read_csv(file_path)
                if df.shape[0] != 0:
                    cursor.execute(CREATE_TABLE_QUERY.format(table_name))

                    column_names = []
                    for column in df.columns:
                        column_name = re.sub(r"\s+", "_", column).lower()  # Replace spaces in column names
                        cursor.execute(ADD_COLUMN_QUERY.format(table_name, column_name + " TEXT"))
                        column_names.append(column_name)
                    
                    # Construct the placeholders for the values
                    placeholders = ", ".join(["%s"] * len(column_names))
                    columns_formatted = ", ".join(column_names)

                    for value in tqdm(df.values, desc=f"Inserting Values into {table_name}"):
                        cursor.execute(INSERT_VALUES_QUERY.format(table_name, columns_formatted, placeholders), tuple(value))

                    print(f'Total baris "{df.shape[0]}", Total kolom "{df.shape[1]}"')
                    print(f'Tabel "{table_name}" berhasil dibuat')
                    print("=".center(100, "="))
                    count_data += 1


        connection.commit()
        print(f'Total Sebanyak {count_data} Tables Data Berhasil Diinput ke Database')
        cursor.close()
        connection.close() 
    except (Exception, psycopg2.DatabaseError) as error:
        print(f'Error: {error}')