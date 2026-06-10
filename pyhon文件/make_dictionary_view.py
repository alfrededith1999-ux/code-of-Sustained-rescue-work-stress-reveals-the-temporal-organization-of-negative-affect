# -*- coding: utf-8 -*-
import sqlite3

db = r"D:\date\psych_master_db_outputs_20260123_104725\psych_master.sqlite"
con = sqlite3.connect(db)

tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
views  = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name").fetchall()]

print("TABLES:", tables)
print("VIEWS :", views)

if "dictionary" not in tables and "dictionary" not in views:
    if "column_dictionary" in tables:
        con.execute("CREATE VIEW dictionary AS SELECT * FROM column_dictionary")
        con.commit()
        print("OK: created VIEW dictionary -> column_dictionary")
    else:
        print("ERROR: no column_dictionary table found, cannot create dictionary view")
else:
    print("No change: dictionary already exists")

con.close()
print("DONE")
