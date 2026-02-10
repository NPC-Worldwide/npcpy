import pandas as pd
import sqlite3
from datetime import datetime

class NPCModelRunner:
    def __init__(self):
        self.compiler = npc_compiler
        self.sql_ops = NPCSQLOperations(npc_compiler)
        self.history_db = os.path.expanduser("~/npcsh_history.db")
        self.model_registry = {}

    def _init_history_db(self):
        with sqlite3.connect(self.history_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_runs (
                    model_name TEXT,
                    run_timestamp DATETIME,
                    run_status TEXT,
                    metadata TEXT,
                    output_preview TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_dependencies (
                    model_name TEXT,
                    depends_on TEXT,
                    created_at DATETIME
                )
            """
            )

    def _load_sample_data(self):
        
        return pd.DataFrame(
            {
                "customer_id": range(1, 4),
                "feedback_text": [
                    "Great service but expensive",
                    "Product needs improvement",
                    "Amazing experience overall",
                ],
                "customer_segment": ["premium", "basic", "premium"],
            }
        )

    def run_model(self, model_name: str, model_sql: str, sample_data=None):
        try:
            
            if sample_data is None:
                raw_data = self._load_sample_data()
            else:
                raw_data = sample_data

            
            self.model_registry["raw_customer_feedback"] = raw_data

            
            if "{{ ref(" in model_sql:
                
                for dep in self._extract_dependencies(model_sql):
                    if dep not in self.model_registry:
                        raise ValueError(f"Dependent model {dep} not found")

                
                model_sql = self._resolve_refs(model_sql)

            
            result_df = self._execute_model(model_sql)

            
            self.model_registry[model_name] = result_df

            
            self._log_model_run(model_name, result_df)

            return result_df

        except Exception as e:
            self._log_model_run(model_name, None, status="failed", error=str(e))
            raise e

    def _execute_model(self, model_sql: str) -> pd.DataFrame:
        
        

        if "synthesize(" in model_sql:
            raw_data = self.model_registry["raw_customer_feedback"]
            return self.sql_ops.synthesize(
                df=raw_data,
                column="feedback_text",
                npc="analyst",
                context="customer_segment",
                framework="satisfaction",
            )

        if "spread_and_sync(" in model_sql:
            input_data = self.model_registry["customer_feedback"]
            return self.sql_ops.spread_and_sync(
                df=input_data,
                column="feedback_analysis",
                npc="strategy_agent",
                variations=["short_term", "long_term"],
                sync_strategy="balanced_analysis",
                context=self._load_vars()["business_context"],
            )

        return pd.DataFrame()  

    def _log_model_run(self, model_name: str, result_df, status="success", error=None):
        with sqlite3.connect(self.history_db) as conn:
            metadata = {
                "status": status,
                "error": error,
                "rows_processed": len(result_df) if result_df is not None else 0,
            }

            preview = result_df.head().to_dict() if result_df is not None else None

            conn.execute(
                """INSERT INTO model_runs
                   (model_name, run_timestamp, run_status, metadata, output_preview)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    model_name,
                    datetime.now().isoformat(),
                    status,
                    json.dumps(metadata),
                    json.dumps(preview),
                ),
            )

    def _load_vars(self):
        with open("npc_project.yml", "r") as f:
            return yaml.safe_load(f).get("vars", {})

    def _extract_dependencies(self, model_sql: str) -> List[str]:
        
        import re

        refs = re.findall(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', model_sql)
        return refs

    def _resolve_refs(self, model_sql: str) -> str:
        
        import re

        def replace_ref(match):
            model_name = match.group(1)
            if model_name in self.model_registry:
                return f"model_{model_name}"
            raise ValueError(f"Model {model_name} not found")

        return re.sub(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', replace_ref, model_sql)

def main():
    

    runner = NPCModelRunner(npc_compiler)

    
    with open("models/customer_feedback.sql", "r") as f:
        feedback_model = runner.run_model("customer_feedback", f.read())
        print("First model results:")
        print(feedback_model.head())

    
    with open("models/customer_insights.sql", "r") as f:
        insights_model = runner.run_model("customer_insights", f.read())
        print("\nSecond model results:")
        print(insights_model.head())

    
    with sqlite3.connect(runner.history_db) as conn:
        history = pd.read_sql(
            "SELECT * FROM model_runs ORDER BY run_timestamp DESC", conn
        )
        print("\nModel run history:")
        print(history)

if __name__ == "__main__":
    main()
