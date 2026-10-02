"""Fresh-process production read probe over a caller-owned synthetic archive."""

import sys
from pathlib import Path

from polylogue.storage.embeddings import identity
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider


def main() -> None:
    root = Path(sys.argv[1])
    identity.EMBEDDING_INPUT_SCHEMA_VERSION += "-relabelled"
    provider = SqliteVecProvider(None, db_path=root / "embeddings.db", archive_root=root, model="voyage-4-lite")
    print(provider.count_session_embeddings(sys.argv[2]))


if __name__ == "__main__":
    main()
