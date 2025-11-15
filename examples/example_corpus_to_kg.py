"""
Example: ingest local files into a corpus and create a knowledge graph with npcpy.

What this shows:
1) Load diverse files (text, CSV, Excel, PDF, DOCX, PPTX, etc.) via npcpy.data.load
2) Build a unified corpus (lightly chunked) from the loaded content
3) Feed the corpus into npcpy.memory.knowledge_graph.kg_initial to create a KG
4) Optionally evolve the KG with new content using kg_evolve_incremental

Notes:
- The KG steps call an LLM (see model/provider args). Defaults are set for a local
  Ollama Qwen variant; change as needed for your environment.
- This example stays light by sampling only a few chunks per file; raise limits
  if you want a deeper ingest.
"""

from pathlib import Path
from typing import Iterable, List

from npcpy.data.load import load_file_contents
from npcpy.memory.knowledge_graph import kg_initial, kg_evolve_incremental


def collect_corpus(
    file_paths: Iterable[Path],
    chunk_size: int = 500,
    max_chunks_per_file: int = 4,
) -> List[str]:
    """Load files and return a list of text chunks suitable for KG ingestion."""
    corpus: List[str] = []
    for path in file_paths:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        try:
            chunks = load_file_contents(str(path), chunk_size=chunk_size)
        except Exception as e:  # keep the pipeline moving
            print(f"Error loading {path}: {e}")
            continue
        if not chunks:
            print(f"No content found in {path}")
            continue
        corpus.extend(chunks[:max_chunks_per_file])
    return corpus


def make_initial_kg(corpus: List[str], model: str, provider: str):
    """Create an initial KG from the corpus text."""
    content = "\n".join(corpus)
    kg = kg_initial(
        content=content,
        model=model,
        provider=provider,
    )
    return kg


def evolve_with_new_files(
    existing_kg,
    new_paths: Iterable[Path],
    model: str,
    provider: str,
    chunk_size: int = 500,
    max_chunks_per_file: int = 2,
):
    """Demonstrate incremental KG evolution with extra documents."""
    new_corpus = collect_corpus(new_paths, chunk_size=chunk_size, max_chunks_per_file=max_chunks_per_file)
    new_text = "\n".join(new_corpus)
    updated_kg = kg_evolve_incremental(
        existing_kg,
        new_content_text=new_text,
        model=model,
        provider=provider,
    )
    return updated_kg


if __name__ == "__main__":
    # Point at files you care about. These sample paths use the repository's test_data folder.
    base_paths = [
        Path("test_data/magical_realism.txt"),   # plain text
        Path("test_data/books.csv"),             # CSV
        Path("test_data/yuan2004.pdf"),          # PDF
        Path("test_data/sample.xlsx"),           # Excel (add your own if missing)
        Path("test_data/sample.docx"),           # Word doc (add your own if missing)
        Path("test_data/sample.pptx"),           # PowerPoint (add your own if missing)
    ]

    corpus = collect_corpus(base_paths, chunk_size=600, max_chunks_per_file=3)
    print(f"Collected {len(corpus)} chunks from {len(base_paths)} files.")

    # Configure your model/provider. Defaults assume an Ollama Qwen model; adjust as needed.
    MODEL = "qwen3:0.6b"
    PROVIDER = "ollama"

    print("\n--- Building initial knowledge graph ---")
    kg = make_initial_kg(corpus, model=MODEL, provider=PROVIDER)
    print(f"KG created with {len(kg.get('facts', []))} facts and {len(kg.get('concepts', []))} concepts.")

    # (Optional) evolve with extra files
    extra_paths = [
        Path("test_data/veo3_video.mp4"),  # will extract audio transcript/metadata if deps available
    ]
    print("\n--- Evolving knowledge graph with additional files ---")
    kg_updated = evolve_with_new_files(kg, extra_paths, model=MODEL, provider=PROVIDER)
    print(
        f"Updated KG: {len(kg_updated.get('facts', []))} facts, "
        f"{len(kg_updated.get('concepts', []))} concepts."
    )
