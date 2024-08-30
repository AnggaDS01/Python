from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()

def connect_to_local_database(
    host=os.getenv('LOCAL_DB_HOST'),
    database=os.getenv('LOCAL_DB_NAME'),
    user=os.getenv('LOCAL_DB_USERNAME'),
    password=os.getenv('LOCAL_DB_PASSWORD')
):

    connection = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )

    return connection

# if __name__ == "__main__":
#     try:
#         connection = connect_to_local_database()
#         print("Successfully connected to the database.")
#         connection.close()
#     except (Exception, psycopg2.DatabaseError) as error:
#         print(f'Error: {error}')