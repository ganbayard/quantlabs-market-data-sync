import mysql.connector
import sshtunnel
import time
import os
from dotenv import load_dotenv

def connect_to_mysql_via_ssh():
    # Load environment variables
    load_dotenv()
    
    # Configuration
    SSH_HOST = os.getenv("SSH_HOST")
    SSH_USER = os.getenv("SSH_USER")
    SSH_KEY_PATH = os.getenv("SSH_KEY_PATH")
    
    # MySQL configuration - with the correct hostname with double 'q'
    DB_HOST = "quantlab-db.c6bqqkgvxujw.eu-central-1.rds.amazonaws.com"
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME", "")
    
    # Create SSH tunnel
    print(f"🔄 Creating SSH tunnel to {SSH_HOST}...")
    
    with sshtunnel.SSHTunnelForwarder(
        (SSH_HOST, 22),
        ssh_username=SSH_USER,
        ssh_pkey=SSH_KEY_PATH,
        remote_bind_address=(DB_HOST, DB_PORT),
        local_bind_address=('127.0.0.1', 3307)
    ) as tunnel:
        print(f"✅ SSH tunnel established on port {tunnel.local_bind_port}")
        
        try:
            # Connect to MySQL through the tunnel
            print(f"🔄 Connecting to MySQL via tunnel...")
            conn = mysql.connector.connect(
                host='127.0.0.1',
                port=tunnel.local_bind_port,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME if DB_NAME else None
            )
            
            # Test the connection
            cursor = conn.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"✅ Connected to MySQL version: {version[0]}")
            
            # List databases
            cursor.execute("SHOW DATABASES")
            databases = cursor.fetchall()
            print("📊 Available databases:")
            for db in databases:
                print(f"  - {db[0]}")
            
            # Here you can execute your actual database operations
            # For example:
            # cursor.execute("SELECT * FROM your_table")
            # results = cursor.fetchall()
            # for row in results:
            #     print(row)
            
            # Close connections
            cursor.close()
            conn.close()
            print("✅ MySQL connection closed")
            
        except Exception as e:
            print(f"❌ MySQL connection error: {e}")

if __name__ == "__main__":
    connect_to_mysql_via_ssh()