
import pyodbc as SQLlibrary
import pandas as pd



class TimeoutException(Exception):
    pass


def download_worker(path, server, database, result_list=None):
    try:
        # Your existing code here
        db_connection = SQLlibrary.connect('Driver={SQL Server};'
                                          'Server=' + server + ';'
                                          'Database=' + database + ';'
                                          'Trusted_Connection=yes;')
        cursor = db_connection.cursor()
        nocount = "SET NOCOUNT ON; "
        ans = "SET ANSI_WARNINGS OFF;"
        result_list.append(db_connection)
        fd = open(path, 'r',encoding='utf-16')
        query=fd.read()
        sql_query1 = nocount+ans+query
        columns = [col_desc[0] for col_desc in cursor.description] if cursor.description else []
        df = pd.read_sql(sql_query1, db_connection, columns=columns)
        result_list.append(df)
    except Exception as e:
        result_list.append(str(e))
    finally:
        if 'db_connection' in locals() and db_connection:
            db_connection.close()


def download(path, server, database, to=60*20):
    result_list = []
    worker_thread = threading.Thread(target=download_worker, args=(path, server, database, result_list))
    worker_thread.start()
    worker_thread.join(timeout=to)
    if worker_thread.is_alive():
        db_connection = result_list[0]
        db_connection.close()
        return 0, "Function call timed out"
    else:
        result = result_list[1]
        return 1, result
