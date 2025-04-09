import mysql.connector
from typing import Any, Dict
from mysql.connector import Error
from config.local_config import LocalConfig
from config.production_config import ProductionConfig

def get_db_connection(config_class: Any) -> mysql.connector.MySQLConnection:
    """
    Create and return a MySQL database connection based on the provided configuration.
    
    Args:
        config_class: Configuration class (LocalConfig or ProductionConfig)
    
    Returns:
        MySQLConnection: A connection to the MySQL database
    
    Raises:
        Error: If connection cannot be established
    """
    try:
        config = config_class()
        connection = mysql.connector.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password
        )
        
        if connection.is_connected():
            print(f"Successfully connected to database: {config.database}")
            return connection
            
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        raise

def execute_query(connection: mysql.connector.MySQLConnection, query: str, params: Dict = None) -> Any:
    """
    Execute a SQL query and return the results.
    
    Args:
        connection: MySQL database connection
        query: SQL query to execute
        params: Dictionary of parameters for the query
    
    Returns:
        Any: Query results
    
    Raises:
        Error: If query execution fails
    """
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, params or {})
        
        if query.strip().upper().startswith('SELECT'):
            result = cursor.fetchall()
        else:
            connection.commit()
            result = cursor.rowcount
            
        cursor.close()
        return result
        
    except Error as e:
        print(f"Error executing query: {e}")
        raise 