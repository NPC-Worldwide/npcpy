from setuptools import setup, find_packages

from pathlib import Path
import os


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths




base_requirements = [
    "jinja2",
    "litellm",    
    "scipy", 
    "numpy",
    "requests",
    "docx", 
    "exa-py", 
    "elevenlabs", 
    "matplotlib",
    "markdown",
    "networkx", 
    "PyYAML",
    "PyMuPDF",
    "pyautogui",
    "pydantic", 
    "pygments",
    "sqlalchemy",
    "termcolor",
    "rich",
    "colorama",
    "docstring_parser",
    "Pillow",
    "python-dotenv",
    "pandas",
    "beautifulsoup4",
    "duckduckgo-search",
    "flask",
    "flask_cors",
    "redis",
    "psycopg2-binary",
    "flask_sse",
    "mcp", 
]


api_requirements = [
    "anthropic",
    "openai",
    "google-generativeai",
    "google-genai",
]


local_requirements = [
    "sentence_transformers",
    "opencv-python",
    "ollama",
    "kuzu",
    "chromadb",
    "diffusers",
    "torch",
    "datasets", 
]


voice_requirements = [
    "pyaudio",
    "gtts",
    "playsound==1.2.2",
    "pygame", 
    "faster_whisper",
    "pyttsx3",
]

extra_files = package_files("npcpy/npc_team/")

setup(
    name="npcpy",
    version="1.3.9",
    packages=find_packages(exclude=["tests*"]),
    install_requires=base_requirements,  
    extras_require={
        "lite": api_requirements,  
        "local": local_requirements,  
        "yap": voice_requirements,  
        "all": api_requirements + local_requirements + voice_requirements,  
    },
    author="Christopher Agostino",
    author_email="info@npcworldwi.de",
    description="npcpy is the premier open-source library for integrating LLMs and Agents into python systems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NPC-Worldwide/npcpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    python_requires=">=3.10",
)

