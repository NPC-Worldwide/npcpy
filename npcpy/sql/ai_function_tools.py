import json
from typing import Dict, Any, List, Optional

class SQLToolCallResponse:
    """
    Represents a structured response with tool calling capabilities 
    that can be generated and processed within SQL
    """
    @staticmethod
    def generate_tool_call_prompt(
        prompt: str, 
        tools: List[Dict[str, Any]], 
        model: str = 'snowflake-arctic'
    ) -> str:
        """
        Generate a prompt that instructs the model to use tools
        
        :param prompt: Original user prompt
        :param tools: List of available tools/functions
        :param model: AI model to use
        :return: Formatted prompt for tool-aware generation
        """
        tool_descriptions = [
            f"Tool: {tool.get('name', 'unnamed')}\n"
            f"Description: {tool.get('description', 'No description')}\n"
            f"Parameters: {json.dumps(tool.get('parameters', {}))}"
            for tool in tools
        ]
        
        return f"""
        You are an AI assistant capable of using the following tools:

        {"\n\n".join(tool_descriptions)}

        User Prompt: {prompt}

        IMPORTANT INSTRUCTIONS:
        1. Carefully analyze the user's request
        2. Determine which tool(s) are most appropriate
        3. Generate a structured JSON response with:
           - tool_calls: List of tool invocations
           - final_response: Your overall response to the user
        4. ONLY use tools that are directly relevant
        5. Format the output as a valid JSON object

        Output Format:
        {{
            "tool_calls": [
                {{
                    "tool_name": "tool_name",
                    "parameters": {{...}}
                }}
            ],
            "final_response": "Optional explanation or summary"
        }}
        """

    @staticmethod
    def parse_tool_calls_sql(tool_call_json: str) -> Dict[str, Any]:
        """
        Parse tool calls within SQL, with error handling
        
        :param tool_call_json: JSON string of tool calls
        :return: Parsed tool call dictionary
        """
        try:
            parsed = json.loads(tool_call_json)
            return {
                'tool_calls': parsed.get('tool_calls', []),
                'final_response': parsed.get('final_response', '')
            }
        except json.JSONDecodeError:
            return {
                'tool_calls': [],
                'final_response': 'Error parsing tool calls'
            }

class SnowflakeSQLToolCaller:
    """
    Snowflake-specific tool calling implementation
    """
    @staticmethod
    def generate_tool_call_sql(
        prompt: str, 
        tools: List[Dict[str, Any]], 
        model: str = 'snowflake-arctic'
    ) -> str:
        """
        Generate a SQL function that performs tool calling
        
        :param prompt: User prompt
        :param tools: Available tools
        :param model: AI model to use
        :return: SQL function definition
        """
        tool_call_prompt = SQLToolCallResponse.generate_tool_call_prompt(
            prompt, tools, model
        )
        
        return f"""
        WITH ai_response AS (
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                model => '{model}',
                prompt => '{tool_call_prompt}'
            ) AS response_json
        ),
        parsed_response AS (
            SELECT 
                response_json,
                PARSE_JSON(response_json) AS parsed_json
            FROM ai_response
        ),
        tool_calls AS (
            SELECT 
                elem.tool_name,
                elem.parameters
            FROM parsed_response,
            LATERAL FLATTEN(input => parsed_json:tool_calls) elem
        )
        SELECT 
            response_json,
            tool_calls.tool_name,
            tool_calls.parameters
        FROM parsed_response
        LEFT JOIN tool_calls ON 1=1
        """

class BigQuerySQLToolCaller:
    """
    BigQuery-specific tool calling implementation
    """
    @staticmethod
    def generate_tool_call_sql(
        prompt: str, 
        tools: List[Dict[str, Any]], 
        model: str = 'text-bison'
    ) -> str:
        """
        Generate a BigQuery ML function for tool calling
        
        :param prompt: User prompt
        :param tools: Available tools
        :param model: AI model to use
        :return: SQL function definition
        """
        tool_call_prompt = SQLToolCallResponse.generate_tool_call_prompt(
            prompt, tools, model
        )
        
        return f"""
        ML.PREDICT(
            MODEL `{model}`,
            (
                SELECT '{tool_call_prompt}' AS prompt
            )
        )
        """

class SQLToolCallOrchestrator:
    """
    Orchestrates tool calling across different SQL databases
    """
    @staticmethod
    def generate_tool_calls(
        engine_type: str,
        prompt: str, 
        tools: List[Dict[str, Any]], 
        model: Optional[str] = None
    ) -> str:
        """
        Generate appropriate SQL for tool calling
        
        :param engine_type: Type of SQL database
        :param prompt: User prompt
        :param tools: Available tools
        :param model: Optional model override
        :return: SQL for tool calling
        """
        model_map = {
            'snowflake': 'snowflake-arctic',
            'bigquery': 'text-bison'
        }
        
        model = model or model_map.get(engine_type.lower(), 'snowflake-arctic')
        
        if engine_type.lower() == 'snowflake':
            return SnowflakeSQLToolCaller.generate_tool_call_sql(
                prompt, tools, model
            )
        elif engine_type.lower() == 'bigquery':
            return BigQuerySQLToolCaller.generate_tool_call_sql(
                prompt, tools, model
            )
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")

# Example integration with ModelCompiler
def _execute_ai_agent_sql(
    self, 
    prompt: str, 
    tools: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute an AI agent entirely within SQL
    
    :param prompt: User prompt
    :param tools: Available tools
    :return: Tool call results
    """
    engine_type = self.engine.dialect.name.lower()
    
    try:
        tool_call_sql = SQLToolCallOrchestrator.generate_tool_calls(
            engine_type, prompt, tools
        )
        
        # Execute the SQL and process results
        df = pd.read_sql(tool_call_sql, self.engine)
        
        # Process tool calls and generate final response
        tool_calls = self._process_sql_tool_calls(df)
        
        return {
            'tool_calls': tool_calls,
            'final_response': df['final_response'].iloc[0] if not df.empty else ''
        }
    
    except Exception as e:
        return {
            'tool_calls': [],
            'final_response': f"Error in SQL tool calling: {str(e)}"
        }

def _process_sql_tool_calls(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process tool calls from SQL result DataFrame
    
    :param df: DataFrame containing tool call results
    :return: List of processed tool calls
    """
    processed_calls = []
    
    for _, row in df.iterrows():
        tool_name = row.get('tool_name')
        parameters = row.get('parameters')
        
        if tool_name and parameters:
            # Execute the tool using existing tool calling mechanism
            tool_result = self._execute_tool(tool_name, parameters)
            
            processed_calls.append({
                'tool_name': tool_name,
                'parameters': parameters,
                'result': tool_result
            })
    
    return processed_calls