import os
import yaml
import sqlalchemy
import pandas as pd
from typing import Dict, Any, Optional

class SQLModelCompiler:
    """
    Compile and execute SQL models across different database engines
    """
    def __init__(
        self, 
        models_dir: str, 
        engine: Optional[sqlalchemy.engine.base.Engine] = None,
        engine_type: str = 'sqlite'
    ):
        """
        Initialize SQL Model Compiler
        
        :param models_dir: Directory containing SQL model files
        :param engine: SQLAlchemy database engine
        :param engine_type: Type of database engine (sqlite, snowflake, bigquery, etc.)
        """
        self.models_dir = models_dir
        self.engine = engine
        self.engine_type = engine_type.lower()
        self.models = {}
        
        # Discover models
        self._discover_models()
    
    def _discover_models(self):
        """
        Discover and load SQL model files
        """
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.sql'):
                model_name = os.path.splitext(filename)[0]
                model_path = os.path.join(self.models_dir, filename)
                
                with open(model_path, 'r') as f:
                    model_content = f.read()
                
                self.models[model_name] = {
                    'name': model_name,
                    'content': model_content,
                    'path': model_path
                }
    
    def _compile_model(self, model_name: str) -> str:
        """
        Compile a SQL model for the specific engine
        
        :param model_name: Name of the model to compile
        :return: Compiled SQL query
        """
        model = self.models[model_name]
        content = model['content']
        
        # Engine-specific compilation
        if self.engine_type == 'snowflake':
            # Snowflake-specific transformations
            content = content.replace('{{', 'SNOWFLAKE.').replace('}}', '')
        elif self.engine_type == 'bigquery':
            # BigQuery-specific transformations
            content = content.replace('{{', 'ML.').replace('}}', '')
        
        return content
    
    def execute_model(
        self, 
        model_name: str, 
        seed_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL model
        
        :param model_name: Name of the model to execute
        :param seed_data: Optional seed data for the model
        :return: Result DataFrame
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Compile model for specific engine
        compiled_sql = self._compile_model(model_name)
        
        # If seed data is provided, prepare the database
        if seed_data and self.engine:
            for table_name, df in seed_data.items():
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        
        # Execute the model
        if self.engine:
            return pd.read_sql(compiled_sql, self.engine)
        else:
            # Fallback to pandas evaluation
            import sqlite3
            
            # Create an in-memory SQLite database for evaluation
            conn = sqlite3.connect(':memory:')
            
            # Load seed data if available
            if seed_data:
                for table_name, df in seed_data.items():
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            return pd.read_sql(compiled_sql, conn)
    
    def run_all_models(self, seed_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Run all discovered models
        
        :param seed_data: Optional seed data for models
        :return: Dictionary of model results
        """
        results = {}
        for model_name in self.models:
            results[model_name] = self.execute_model(model_name, seed_data)
        return results

# Example usage in a pipeline
def create_model_compiler(
    models_dir: str, 
    engine_type: str = 'sqlite', 
    connection_params: Optional[Dict[str, Any]] = None
) -> SQLModelCompiler:
    """
    Create a SQL Model Compiler with the specified engine
    
    :param models_dir: Directory containing SQL model files
    :param engine_type: Type of database engine
    :param connection_params: Connection parameters for the database
    :return: SQLModelCompiler instance
    """
    if engine_type == 'snowflake':
        from sqlalchemy.dialects.snowflake import base
        engine = sqlalchemy.create_engine(
            f"snowflake://{connection_params['username']}:{connection_params['password']}@"
            f"{connection_params['account']}/{connection_params['database']}/{connection_params['schema']}"
        )
    elif engine_type == 'bigquery':
        from google.cloud import bigquery
        from sqlalchemy.dialects.bigquery import base
        engine = sqlalchemy.create_engine(
            f"bigquery://{connection_params['project_id']}"
        )
    else:
        # Default to SQLite
        engine = sqlalchemy.create_engine('sqlite:///models.db')
    
    return SQLModelCompiler(
        models_dir=models_dir, 
        engine=engine, 
        engine_type=engine_type
    )