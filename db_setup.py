import sqlite3

DB_PATH = "doctors.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Create doctors table
c.execute("""
CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

conn.commit()
conn.close()
print("Database initialized successfully!")