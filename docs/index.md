<p align="center">
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc_team/npcsh_sibiji.png" alt="npcsh logo with sibiji the spider">
</p>


# npcsh


- `npcsh` is a python-based AI Agent framework designed to integrate Large Language Models (LLMs) and Agents into one's daily workflow by making them available and easily configurable through a command line shell as well as an extensible python library.

- **Smart Interpreter**: `npcsh` leverages the power of LLMs to understand your natural language commands and questions, executing tasks, answering queries, and providing relevant information from local files and the web.

- **Macros**: `npcsh` provides macros to accomplish common tasks with LLMs like voice control (`/whisper`), image generation (`/vixynt`), screenshot capture and analysis (`/ots`), one-shot questions (`/sample`), computer use (`/plonk`),  retrieval augmented generation (`/rag`), search (`/search`) and more. Users can also build their own jinxs and call them like macros from the shell.


- **NPC-Driven Interactions**: `npcsh` allows users to take advantage of agents (i.e. NPCs) through a managed system. Users build a directory of NPCs and associated jinxs that can be used to accomplish complex tasks and workflows. NPCs can be tailored to specific tasks and have unique personalities, directives, and jinxs. Users can combine NPCs and jinxs in assembly line like workflows or use them in SQL-style models.

* **Extensible with Python:**  `npcsh`'s python package provides useful functions for interacting with LLMs, including explicit coverage for popular providers like ollama, anthropic, openai, gemini, deepseek, and openai-like providers. Each macro has a corresponding function and these can be used in python scripts. `npcsh`'s functions are purpose-built to simplify NPC interactions but NPCs are not required for them to work if you don't see the need.

* **Simple, Powerful CLI:**  Use the `npc` CLI commands to run `npcsh` macros or commands from one's regular shell. Set up a flask server so you can expose your NPC team for use as a backend service. You can also use the `npc` CLI to run SQL models defined in your project, execute assembly lines, and verify the integrity of your NPC team's interrelations. `npcsh`'s NPCs take advantage of jinja templating to reference other NPCs and jinxs in their properties, and the `npc` CLI can be used to verify these references.

* **Powerful jinx integrations:** `npcsh` has built-in jinxs for users to have agents execute code, analyze data, generate images, search the web, and more. jinxs can be defined in YAML files as part of project-specific `npc_team`s or in the global `~/.npcsh/npc_team` directory or simply in python scripts. Once compiled, the jinxs can be used as macros in the `npc` cli as well as `/{jinx_name}` commands in the `npcsh` shell.



Interested to stay in the loop and to hear the latest and greatest about `npcsh` ? Be sure to sign up for the [npcsh newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)!



## Quick Links
- [Installation Guide](installation.md)
- [NPC Shell User Guide](guide.md)
- [NPC Data Layer](npc_data_layer.md)
- [TLDR Cheat Sheet](TLDR_Cheat_sheet.md)
- [API Reference](api/index.md)





## Contributing
Contributions are welcome! Please submit issues and pull requests on the GitHub repository.

## Support
If you appreciate the work here, [consider supporting NPC Worldwide](https://buymeacoffee.com/npcworldwide). If you'd like to explore how to use `npcsh` to help your business, please reach out to info@npcworldwi.de .


## NPC Studio
Coming soon! NPC Studio will be a desktop application for managing chats and agents on your own machine.
Be sure to sign up for the [npcsh newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A) to hear updates!

## License
This project is licensed under the MIT License.

