import sqlalchemy
from concurrent.futures import ThreadPoolExecutor


def bulk_insert_data(df, server, database, chunk_size=1000, env_type='dev', num_threads=8):
    connection_params = {
        'driver': 'SQL Server',
        'server': server,
        'database': database,
        'trusted_connection': 'yes'
    }

    connection_string = ";".join([f"{key}={value}" for key, value in connection_params.items()])
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")

    # Insert Dataframe into SQL Server in batches using parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            future = executor.submit(insert_chunk, chunk_df, engine, env_type)
            futures.append(future)
            # Wait for all tasks to complete
        for future in futures:
            future.result()
    print("Data has been inserted successfully into the database.")


def insert_chunk(chunk_df, engine, env_type):
    table_name = 'SCORING_DEV' if env_type == 'dev' else 'SCORING'
    chunk_df.to_sql(table_name, con=engine, schema='MODELS', index=False, if_exists='append')
