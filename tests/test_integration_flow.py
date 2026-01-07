import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from polylogue.config import Source, default_config, write_config
from polylogue.lib.repository import ConversationRepository
from polylogue.run import run_sources
from polylogue.server.app import app
from polylogue.server.deps import get_repository


@pytest.fixture
def sample_data():
    return Path(__file__).parent / "samples" / "complex_chatgpt.json"


def test_end_to_end_flow(workspace_env, tmp_path, sample_data):
    """
    Test the full lifecycle:
    1. Ingest realistic source
    2. Run pipeline (ingest -> store -> render -> search index)
    3. Verify via Web UI and API
    """
    if not sample_data.exists():
        pytest.skip("Sample data not found")

    # 1. Setup Source
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    shutil.copy(sample_data, inbox / "export.json")

    config = default_config()
    config.sources = [Source(name="test_inbox", path=inbox)]
    write_config(config)

    # 2. Run Pipeline
    run_sources(config=config, stage="all")

    # 3. Verify Store & Index
    db_path = workspace_env["state_root"] / "polylogue.db"
    assert db_path.exists()

    repo = ConversationRepository(db_path)

    # Verify Search
    hits = repo.search("quick sort")
    assert len(hits) == 1
    assert hits[0].title == "Complex Conversation With Markdown"

    # 4. Verify API & UI Integration
    # Override app dependency to use the test DB we just populated
    def override_repo():
        return ConversationRepository(db_path)

    app.dependency_overrides[get_repository] = override_repo
    client = TestClient(app)

    # API List
    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    convs = resp.json()
    assert len(convs) >= 1
    cid = convs[0]["id"]

    # API Detail
    resp = client.get(f"/api/conversations/{cid}")
    assert resp.status_code == 200
    data = resp.json()
    assert "divide-and-conquer" in data["messages"][1]["text"]

    # Web View (Rendered HTML check)
    resp = client.get(f"/view/{cid}")
    assert resp.status_code == 200
    content = resp.text

    # Check for Markdown features rendered
    assert "<table" in content or "<table>" in content  # basic check for table rendering
    assert "quick_sort" in content  # code block content

    app.dependency_overrides.clear()
