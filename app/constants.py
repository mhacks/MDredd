import os

DB_FILE = os.getenv("DB_FILE", "mdredd.db")
SNAPSHOT_INTERVAL = 10  # in number of requests
