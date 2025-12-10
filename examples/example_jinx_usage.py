import os
import yaml
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

# --- Assumed Imports from your npcpy installation ---
# This script assumes 'npcpy' is installed and its components are accessible.
# If these imports fail, the script will inform you to install npcpy.
try:
    from npcpy.npc_compiler import Jinx, NPC
    from jinja2 import Environment, DictLoader, Undefined as SilentUndefined
    print("--- Successfully imported Jinx, NPC, and Jinja2 from installed libraries. ---")
except ImportError:
    print("\n--- ERROR: Could not import Jinx/NPC from npcpy or jinja2. ---")
    print("--- Please ensure 'npcpy' is installed (e.g., 'pip install npcpy') ---")
    print("--- and that jinja2 is available. This script cannot run without them. ---")
    print("--- Exiting demonstration. ---")
    exit(1) # Exit if essential components are not found

# --- Minimal Example: In-Memory Jinx ---
def run_in_memory_jinx_demo():
    print("\n--- Running In-Memory Jinx Demo ---")

    # 1. Define the Jinx's YAML data as a Python dictionary.
    #    Notice the Jinja {% for %} and {% if %} directly in the 'steps' list.
    jinx_data = {
        "jinx_name": "readme_in_memory_jinx",
        "description": "A Jinx defined in-memory demonstrating Jinja structural looping and conditional steps.",
        "inputs": [
            {"num_tasks": 2},
            {"include_greeting": True},
            {"processing_items": ["apple", "banana"]}
        ],
        "steps": [
            "{% if include_greeting %}",
            "- name: greet_start",
            "  engine: python",
            "  code: |",
            "    output = 'Hello from in-memory Jinx!'",
            "    context['greeting_message'] = output",
            "{% endif %}",
            "",
            "{% for i in range(num_tasks) %}",
            "- name: process_task_{{ i + 1 }}",
            "  engine: python",
            "  code: |",
            "    task_index = {{ i }}",
            "    current_item = context['processing_items'][task_index] if task_index < len(context['processing_items']) else 'N/A'",
            "    output = f'Processing task {{ i + 1 }} for item: {current_item}'",
            "    context['last_processed_item'] = current_item",
            "{% endfor %}",
            "",
            "- name: final_report",
            "  engine: python",
            "  code: |",
            "    greeting = context.get('greeting_message', 'No greeting.')",
            "    last_item = context.get('last_processed_item', 'Nothing processed.')",
            "    output = f'Report:\\n- {greeting}\\n- Last item processed: {last_item}\\n- Total tasks expected: {{ num_tasks }}'",
        ]
    }

    # 2. Create a Jinja Environment for the first-pass rendering.
    #    This environment needs access to the Jinx's 'inputs' to resolve structural Jinja.
    jinja_env_for_first_pass = Environment(
        loader=DictLoader({}), # No external templates needed for this env
        undefined=SilentUndefined, # Handle undefined variables gracefully during first pass
    )
    
    # Extract input defaults/values for Jinja globals during first pass
    # This simulates how the Jinx framework would make these available.
    for inp in jinx_data["inputs"]:
        if isinstance(inp, dict):
            key = list(inp.keys())[0]
            jinja_env_for_first_pass.globals[key] = inp[key]
    
    # 3. Instantiate the Jinx.
    my_jinx = Jinx(jinx_data=jinx_data)

    # 4. Perform the first-pass rendering to expand Jinja structural elements.
    print("\n--- Performing First-Pass Rendering (Structural Templating) ---")
    my_jinx.render_first_pass(jinja_env_for_first_pass, {}) # all_jinx_callables is empty for this demo
    
    print("\n--- Rendered Steps (after first pass): ---")
    print(yaml.dump(my_jinx.steps, default_flow_style=False))

    # 5. Prepare an NPC and input values for execution.
    #    The NPC's shared_context will be used by the Jinx.
    mock_npc = NPC(name="LAVANZARO")
    runtime_input_values = {
        "num_tasks": 2,
        "include_greeting": True,
        "processing_items": ["alpha", "beta", "gamma"] # These override defaults from jinx_data for execution
    }

    # 6. Execute the Jinx.
    print("\n--- Executing Jinx with runtime inputs: ---")
    print(f"Runtime Inputs: {runtime_input_values}")
    final_context = my_jinx.execute(
        input_values=runtime_input_values,
        npc=mock_npc,
        jinja_env=Environment(loader=DictLoader({}), undefined=SilentUndefined) # A separate Jinja env for runtime code rendering
    )

    # 7. Print the final output.
    print("\n--- Jinx Execution Complete ---")
    print("\nFinal Output:")
    print(final_context.get('output'))
    print("\nFinal Context (relevant parts):")
    # Access outputs by step name, as set in the Jinx's execution logic
    print(f"  Greeting Message (from 'greet_start' step): {final_context.get('greet_start')}")
    print(f"  Last Processed Item (from 'process_task_2' step output): {final_context.get('process_task_2')}")
    print(f"  Final Report Output (from 'final_report' step): {final_context.get('final_report')}")


# --- Minimal Example: File-Based Jinx ---
def run_file_based_jinx_demo():
    print("\n--- Running File-Based Jinx Demo ---")

    # 1. Define Jinx data for the file.
    file_jinx_data = {
        "jinx_name": "readme_file_jinx",
        "description": "A Jinx loaded from a file demonstrating Jinja structural looping and conditional steps.",
        "inputs": [
            {"num_loops": 2},
            {"show_extra_info": False},
            {"data_points": ["data1", "data2"]}
        ],
        "steps": [
            "{% if show_extra_info %}",
            "- name: show_info",
            "  engine: python",
            "  code: |",
            "    output = 'Extra information is enabled!'",
            "    context['info_status'] = 'enabled'",
            "{% else %}",
            "- name: hide_info",
            "  engine: python",
            "  code: |",
            "    output = 'Extra information is disabled.'",
            "    context['info_status'] = 'disabled'",
            "{% endif %}",
            "",
            "- name: start_processing",
            "  engine: python",
            "  code: |",
            "    context['processed_count'] = 0",
            "    output = 'Starting data processing.'",
            "",
            "{% for k in range(num_loops) %}",
            "- name: process_data_loop_{{ k + 1 }}",
            "  engine: python",
            "  code: |",
            "    context['processed_count'] += 1",
            "    current_data = context['data_points'][{{ k }}] if {{ k }} < len(context['data_points']) else 'N/A'",
            "    output = f'Loop {{ k + 1 }}: Processing {current_data}'",
            "{% endfor %}",
            "",
            "- name: final_report",
            "  engine: python",
            "  code: |",
            "    output = f'Report: Info status: {{ context[\"info_status\"] }}. Total items processed: {{ context[\"processed_count\"] }}.'",
        ]
    }

    # 2. Create a temporary .jinx file.
    #    This simulates loading from a file system.
    temp_dir = tempfile.mkdtemp()
    jinx_file_path = os.path.join(temp_dir, "file_demo_jinx.jinx")
    
    # Write the Jinx data to the temporary file
    with open(jinx_file_path, 'w') as f:
        yaml.dump(file_jinx_data, f, default_flow_style=False)
    print(f"Created temporary Jinx file at: {jinx_file_path}")

    # 3. Create a Jinja Environment for first-pass rendering.
    jinja_env_for_first_pass_file = Environment(
        loader=DictLoader({}),
        undefined=SilentUndefined,
    )
    # Extract input defaults/values for Jinja globals during first pass
    for inp in file_jinx_data["inputs"]:
        if isinstance(inp, dict):
            key = list(inp.keys())[0]
            jinja_env_for_first_pass_file.globals[key] = inp[key]

    # 4. Load the Jinx from the file path.
    my_jinx_from_file = Jinx(jinx_path=jinx_file_path)

    # 5. Perform the first-pass rendering.
    print("\n--- Performing First-Pass Rendering (Structural Templating) for File Jinx ---")
    my_jinx_from_file.render_first_pass(jinja_env_for_first_pass_file, {})
    print("\n--- Rendered Steps After First Pass for File Jinx ---")
    print(yaml.dump(my_jinx_from_file.steps, default_flow_style=False))

    # 6. Execute the Jinx.
    print("\n--- Executing File Jinx with runtime inputs: ---")
    mock_npc_file = NPC(name="ETNA")
    runtime_input_values_file = {
        "num_loops": 2,
        "show_extra_info": False,
        "data_points": ["alpha", "beta"]
    }
    print(f"Runtime Inputs: {runtime_input_values_file}")
    final_context_file = my_jinx_from_file.execute(
        input_values=runtime_input_values_file,
        npc=mock_npc_file,
        jinja_env=Environment(loader=DictLoader({}), undefined=SilentUndefined)
    )

    # 7. Print the final output.
    print("\n--- File Jinx Execution Complete ---")
    print("\nFinal Output:")
    print(final_context_file.get("output"))
    print("\nFinal Context (relevant parts):")
    print(f"  Info Status (from 'hide_info' step): {final_context_file.get('hide_info')}")
    print(f"  Final Report Output (from 'final_report' step): {final_context_file.get('final_report')}")

    # 8. Clean up the temporary file and directory.
    os.remove(jinx_file_path)
    os.rmdir(temp_dir)
    print(f"\nCleaned up temporary Jinx file and directory: {temp_dir}")


if __name__ == "__main__":
    run_in_memory_jinx_demo()
    print("\n" + "="*80 + "\n") # Separator
    run_file_based_jinx_demo()
    print("\n--- All Demos Finished ---")