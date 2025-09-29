from typing import Dict, Callable, Any

class DatabaseAIFunctionMapper:
    @staticmethod
    def get_snowflake_cortex_mapping() -> Dict[str, Dict[str, Any]]:
        """
        Map NPC AI functions to Snowflake Cortex functions
        
        Structure:
        {
            'npc_function_name': {
                'cortex_function': 'snowflake_cortex_function_name',
                'transformer': optional_transformation_function
            }
        }
        """
        return {
            # Text Generation Mapping
            'generate_text': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda prompt, **kwargs: f"SNOWFLAKE.CORTEX.COMPLETE('{prompt}')"
            },
            
            # Summarization Mapping
            'summarize': {
                'cortex_function': 'SUMMARIZE',
                'transformer': lambda text, **kwargs: f"SNOWFLAKE.CORTEX.SUMMARIZE('{text}')"
            },
            
            # Sentiment Analysis Mapping
            'analyze_sentiment': {
                'cortex_function': 'SENTIMENT',
                'transformer': lambda text, **kwargs: f"SNOWFLAKE.CORTEX.SENTIMENT('{text}')"
            },
            
            # Translation Mapping
            'translate': {
                'cortex_function': 'TRANSLATE',
                'transformer': lambda text, source_lang='auto', target_lang='en', **kwargs: 
                    f"SNOWFLAKE.CORTEX.TRANSLATE('{text}', '{source_lang}', '{target_lang}')"
            },
            
            # Named Entity Recognition
            'extract_entities': {
                'cortex_function': 'EXTRACT_ENTITIES',
                'transformer': lambda text, **kwargs: f"SNOWFLAKE.CORTEX.EXTRACT_ENTITIES('{text}')"
            },
            
            # Embedding Generation
            'generate_embedding': {
                'cortex_function': 'EMBED_TEXT',
                'transformer': lambda text, model='snowflake-arctic', **kwargs: 
                    f"SNOWFLAKE.CORTEX.EMBED_TEXT('{model}', '{text}')"
            }
        }
    
    @staticmethod
    def get_databricks_ai_mapping() -> Dict[str, Dict[str, Any]]:
        """
        Map NPC AI functions to Databricks AI functions
        """
        return {
            # Databricks uses different function names and approaches
            'generate_text': {
                'databricks_function': 'serving.predict',
                'transformer': lambda prompt, model='databricks-dolly', **kwargs: 
                    f"serving.predict('{model}', '{prompt}')"
            },
            # Add more Databricks-specific mappings
        }
    
    @staticmethod
    def get_bigquery_ai_mapping() -> Dict[str, Dict[str, Any]]:
        """
        Map NPC AI functions to BigQuery AI functions
        """
        return {
            'generate_text': {
                'bigquery_function': 'ML.GENERATE_TEXT',
                'transformer': lambda prompt, model='text-bison', **kwargs:
                    f"ML.GENERATE_TEXT(MODEL `{model}`, '{prompt}')"
            },
            # Add more BigQuery-specific mappings
        }

class NativeDatabaseAITransformer:
    def __init__(self, database_type: str):
        self.database_type = database_type
        self.function_mappings = self._get_database_mappings()
    
    def _get_database_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the appropriate AI function mapping based on database type
        """
        mappings = {
            'snowflake': DatabaseAIFunctionMapper.get_snowflake_cortex_mapping(),
            'databricks': DatabaseAIFunctionMapper.get_databricks_ai_mapping(),
            'bigquery': DatabaseAIFunctionMapper.get_bigquery_ai_mapping()
        }
        return mappings.get(self.database_type.lower(), {})
    
    def transform_ai_function(self, function_name: str, **kwargs) -> str:
        """
        Transform an NPC AI function to a native database AI function
        """
        mapping = self.function_mappings.get(function_name)
        if not mapping:
            raise ValueError(f"No native mapping found for function: {function_name}")
        
        transformer = mapping.get('transformer')
        if not transformer:
            raise ValueError(f"No transformer found for function: {function_name}")
        
        return transformer(**kwargs)

# Example usage in ModelCompiler
def _has_native_ai_functions(self, source_name: str) -> bool:
    """Enhanced method to check native AI function support"""
    ai_enabled = {
        'snowflake': True,
        'databricks': True,
        'bigquery': True
    }
    return ai_enabled.get(source_name.lower(), False)

def _execute_ai_model(self, sql: str, model: SQLModel) -> pd.DataFrame:
    """
    Enhanced method to use native AI functions when available
    """
    source_pattern = r'FROM\s+(\\w+)\\.(\\w+)'
    matches = re.findall(source_pattern, sql)
    
    if matches:
        source_name, table_name = matches[0]
        engine = self._get_engine(source_name)
        
        # Check for native AI function support
        if self._has_native_ai_functions(source_name):
            # Use native transformer
            transformer = NativeDatabaseAITransformer(source_name)
            
            # Modify SQL to use native AI functions
            for func_name, params in model.ai_functions.items():
                try:
                    native_func_call = transformer.transform_ai_function(
                        func_name, 
                        text=params.get('column', ''),
                        **{k: v for k, v in params.items() if k != 'column'}
                    )
                    
                    # Replace the NQL function with native function
                    sql = sql.replace(
                        f"nql.{func_name}({params.get('column', '')})", 
                        native_func_call
                    )
                except ValueError as e:
                    # Fallback to original method if transformation fails
                    print(f"Warning: {e}. Falling back to default AI function.")
            
            return pd.read_sql(sql.replace(f"{source_name}.", ""), engine)
    
    # Fallback to existing AI model execution
    return super()._execute_ai_model(sql, model)