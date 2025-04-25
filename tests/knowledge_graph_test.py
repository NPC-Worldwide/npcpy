from npcpy.memory.knowledge_graph import *
import os

# Example usage:
if __name__ == "__main__":
    db_path = os.path.expanduser("~/npcsh_graph.db")  # Specify your database path here
    text = """
npcsh is a python based command line tool designed to integrate Large Language Models (LLMs) into one's daily workflow by making them available through the command line shell.

Smart Interpreter: npcsh leverages the power of LLMs to understand your natural language commands and questions, executing tasks, answering queries, and providing relevant information from local files and the web.

Macros: npcsh provides macros to accomplish common tasks with LLMs like voice control (/yap), image generation (/vixynt), screenshot capture and analysis (/ots), one-shot questions (/sample), and more.

NPC-Driven Interactions: npcsh allows users to coordinate agents (i.e. NPCs) to form assembly lines that can reliably accomplish complicated multi-step procedures. Define custom "NPCs" (Non-Player Characters) with specific personalities, directives, and tools. This allows for tailored interactions based on the task at hand.

Tool Use: Define custom tools for your NPCs to use, expanding their capabilities beyond simple commands and questions. Some example tools include: image generation, local file search, data analysis, web search, local file search, bash command execution, and more.

Extensible with Python: Write your own tools and extend npcsh's functionality using Python or use our functionis to simplify interactions with LLMs in your projects.

Bash Wrapper: Execute bash commands directly without leaving the shell. Use your favorite command-line tools like VIM, Emacs, ipython, sqlite3, git, and more without leaving the shell!


    """
    path = os.path.expanduser("~/npcww/npcsh/tests/")
    conn = init_db(db_path, drop=False)

    facts = process_text(
        db_path, text, path, model="gpt-4o-mini", provider="openai", conn=conn
    )
    groups = identify_groups(facts, model="gpt-4o-mini", provider="openai")
    print("\nIdentified Groups:")

    for group in groups:
        print(f"- {group}")
        create_group(conn, group)

    # For each fact, assign it to appropriate groups

    for fact in facts:
        group_assignments = assign_to_groups(
            fact, groups, model="gpt-4o-mini", provider="openai"
        )
        for group in group_assignments["groups"]:
            print(group)
            assign_fact_to_group(conn, fact, group)

    conn.close()


## ultimately wwell do the vector store in the main db. so when we eventually starti adding new facts well  do so by checking similar facts
# there and then if were doing the rag search well do a rag and then graph
