"""
Standalone ingestion script — (re)loads the knowledge base into ChromaDB.

Usage:
    python -m scripts.ingest

This is the same ingestion that runs automatically on FastAPI startup,
but can be run manually if you want to update the knowledge base without
restarting the server.
"""

import sys
from pathlib import Path

# Add project root to Python path so we can import the app module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.rag import ingest_knowledge_base


def main():
    print("🔄 Re-ingesting knowledge base into ChromaDB...")
    ingest_knowledge_base()
    print("✅ Done! Knowledge base is up to date.")


if __name__ == "__main__":
    main()
