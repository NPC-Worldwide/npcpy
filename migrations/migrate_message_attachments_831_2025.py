import sqlite3
import os

db_path = os.path.expanduser(os.environ.get('NPCSH_DB_PATH', '~/npcsh_history.db'))

def migrate_database(db_path):
    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("Checking for 'file_path' column in 'message_attachments' table...")
        cursor.execute("PRAGMA table_info(message_attachments)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'file_path' not in columns:
            print("'file_path' column not found. Adding it now...")
            cursor.execute("ALTER TABLE message_attachments ADD COLUMN file_path TEXT")
            print("Column 'file_path' added successfully.")
        else:
            print("'file_path' column already exists.")
        
        conn.commit()
        print("Migration complete and changes committed.")
        
    except sqlite3.Error as e:
        print(f"An error occurred during migration: {e}")
        conn.rollback()
    finally:
        conn.close()
        print("Database connection closed.")

if __name__ == '__main__':
    if os.path.exists(db_path):
        migrate_database(db_path)
    else:
        print(f"Database file not found at {db_path}. Please ensure the path is correct.")