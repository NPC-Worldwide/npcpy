import sqlalchemy
from typing import Dict, Any, Optional, Callable
import textwrap

class DatabaseAIAdapter:
    """
    Base class for database-specific AI function adapters
    """
    def __init__(self, engine: sqlalchemy.engine.base.Engine):
        self.engine = engine
        self.dialect = self._get_dialect()

    def _get_dialect(self) -> str:
        """Determine the specific database dialect"""
        dialect_map = {
            'postgresql': 'postgresql',
            'mysql': 'mysql',
            'mssql': 'mssql',
            'sqlite': 'sqlite',
            'snowflake': 'snowflake'
        }
        return dialect_map.get(self.engine.dialect.name.lower(), 'unknown')

    def generate_ai_function(self, function_type: str, prompt: str, **kwargs) -> str:
        """
        Generate AI function implementation based on database type
        
        :param function_type: Type of AI function (generate_text, summarize, etc.)
        :param prompt: Input prompt
        :param kwargs: Additional parameters
        :return: SQL implementation of AI function
        """
        adapter_method = getattr(self, f'_{self.dialect}_{function_type}', None)
        
        if adapter_method:
            return adapter_method(prompt, **kwargs)
        
        # Fallback to generic implementation
        return self._generic_ai_function(function_type, prompt, **kwargs)

    def _generic_ai_function(self, function_type: str, prompt: str, **kwargs) -> str:
        """
        Generic fallback implementation using Python-based AI processing
        """
        # Create a temporary table-based approach for AI function simulation
        return textwrap.dedent(f'''
        WITH ai_input AS (
            SELECT '{prompt}' AS input_text
        )
        SELECT 
            CASE 
                WHEN '{function_type}' = 'generate_text' THEN 
                    'Generated text based on: ' || input_text
                WHEN '{function_type}' = 'summarize' THEN 
                    'Summary of: ' || input_text
                WHEN '{function_type}' = 'analyze_sentiment' THEN 
                    CASE 
                        WHEN input_text LIKE '%good%' OR input_text LIKE '%great%' THEN 'positive'
                        WHEN input_text LIKE '%bad%' OR input_text LIKE '%terrible%' THEN 'negative'
                        ELSE 'neutral'
                    END
                ELSE input_text
            END AS ai_result
        FROM ai_input
        ''')

    def _postgresql_generate_text(self, prompt: str, **kwargs) -> str:
        """
        PostgreSQL-specific text generation using pgai extension
        Requires: CREATE EXTENSION IF NOT EXISTS pgai;
        """
        return textwrap.dedent(f'''
        SELECT pgai.generate_text(
            model => 'openai-gpt-3.5-turbo', 
            prompt => '{prompt}'
        ) AS generated_text
        ''')

    def _mysql_generate_text(self, prompt: str, **kwargs) -> str:
        """
        MySQL-specific text generation 
        Uses a custom table-based approach with external AI call simulation
        """
        return textwrap.dedent(f'''
        WITH ai_input AS (
            SELECT '{prompt}' AS input_text
        )
        SELECT 
            CONCAT('Generated text based on: ', input_text) AS generated_text
        FROM ai_input
        ''')

    def _mssql_generate_text(self, prompt: str, **kwargs) -> str:
        """
        MSSQL-specific text generation
        Uses a CLR integration or external call simulation
        """
        return textwrap.dedent(f'''
        WITH ai_input AS (
            SELECT '{prompt}' AS input_text
        )
        SELECT 
            CONCAT('Generated text based on: ', input_text) AS generated_text
        FROM ai_input
        ''')

    def _postgresql_summarize(self, text: str, **kwargs) -> str:
        """PostgreSQL summarization implementation"""
        return textwrap.dedent(f'''
        SELECT pgai.summarize(
            text => '{text}',
            max_length => 100
        ) AS summary
        ''')

    def _postgresql_analyze_sentiment(self, text: str, **kwargs) -> str:
        """PostgreSQL sentiment analysis implementation"""
        return textwrap.dedent(f'''
        SELECT 
            CASE 
                WHEN pgai.sentiment_score('{text}') > 0 THEN 'positive'
                WHEN pgai.sentiment_score('{text}') < 0 THEN 'negative'
                ELSE 'neutral'
            END AS sentiment
        ''')

class AIFunctionRouter:
    """
    Routes AI function calls to appropriate database-specific adapters
    """
    @staticmethod
    def route_ai_function(engine: sqlalchemy.engine.base.Engine, 
                           function_type: str, 
                           prompt: str, 
                           **kwargs) -> str:
        """
        Route AI function to appropriate database adapter
        
        :param engine: SQLAlchemy database engine
        :param function_type: Type of AI function
        :param prompt: Input prompt
        :param kwargs: Additional parameters
        :return: SQL implementation of AI function
        """
        adapter = DatabaseAIAdapter(engine)
        return adapter.generate_ai_function(function_type, prompt, **kwargs)

# Example integration with existing ModelCompiler
def _execute_ai_model(self, sql: str, model: SQLModel) -> pd.DataFrame:
    """
    Enhanced method to use AI function adapters
    """
    from npcpy.sql.database_ai_adapters import AIFunctionRouter
    
    # Existing code to determine source and engine
    source_pattern = r'FROM\s+(\\w+)\\.(\\w+)'
    matches = re.findall(source_pattern, sql)
    
    if matches:
        source_name, table_name = matches[0]
        engine = self._get_engine(source_name)
        
        # Modify SQL to use database-specific AI functions
        for func_name, params in model.ai_functions.items():
            try:
                # Route AI function through adapter
                native_func_call = AIFunctionRouter.route_ai_function(
                    engine,
                    func_name, 
                    text=params.get('column', ''),
                    **{k: v for k, v in params.items() if k != 'column'}
                )
                
                # Replace the NQL function with native/adapted function
                sql = sql.replace(
                    f"nql.{func_name}({params.get('column', '')})", 
                    native_func_call
                )
            except Exception as e:
                # Fallback to original method if transformation fails
                print(f"Warning: AI function adaptation failed: {e}. Falling back to default.")
        
        return pd.read_sql(sql.replace(f"{source_name}.", ""), engine)
    
    # Fallback to existing AI model execution
    return super()._execute_ai_model(sql, model)