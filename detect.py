import os, psutil
from lib.datamanagertest import DataManagerTest
import argparse
import psycopg2
import traceback

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='file', type=str, help='Image file path to find cetacean fin')
    parser.add_argument('--conn_string', dest='conn_string', type=str, help='Database string connection')
    parser.add_argument('--table_name', dest='table_name', type=str, help='Table to execute update operation')
    parser.add_argument('--record_id', dest='record_id', type=str, help='Record id to data update')
    args = parser.parse_args()
    file = args.file
    conn_string = args.conn_string
    table_name = args.table_name
    record_id = args.record_id

    con = psycopg2.connect(conn_string)
    cur = con.cursor()

    manager = DataManagerTest(tf_path='/var/www/html/tensorflow/')
    result = manager.runInternal(file)
    sql = "UPDATE " + table_name + " SET status='finished', processing_date = now() WHERE id = " + record_id

    if (result):
        sql = "UPDATE " + table_name + " SET processed_data = '" + str(result).replace("'", '"') + "', status='finished', processing_date = now() WHERE id = " + record_id
    
    cur.execute(sql)
    con.commit()

    con.close()

    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss) 

    print(str(result).replace("'", '"'))
    
except Exception:
    print(
        traceback.format_exc()
    )