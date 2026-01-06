
from unittest.mock import MagicMock, patch

import pytest

from polylogue.index_qdrant import QdrantError, VectorStore, get_embeddings, update_qdrant_for_conversations


@patch("polylogue.index_qdrant.httpx.Client")
@patch.dict("os.environ", {"VOYAGE_API_KEY": "fake-key"})
def test_get_embeddings(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
    mock_client.post.return_value = mock_response

    embeddings = get_embeddings(["text"])
    assert embeddings == [[0.1, 0.2]]
    mock_client.post.assert_called_once()

@patch.dict("os.environ", {}, clear=True)
def test_get_embeddings_no_key():
    with pytest.raises(QdrantError, match="VOYAGE_API_KEY"):
        get_embeddings(["text"])

@patch("polylogue.index_qdrant.QdrantClient")
@patch("polylogue.index_qdrant.get_embeddings")
def test_upsert_messages(mock_get_embeddings, mock_qdrant_cls):
    mock_client = MagicMock()
    mock_qdrant_cls.return_value = mock_client
    mock_get_embeddings.return_value = [[0.1, 0.2]]

    store = VectorStore(api_key="test")
    messages = [{"message_id": "m1", "conversation_id": "c1", "provider_name": "p1", "content": "text"}]
    
    store.upsert_messages(messages)
    
    mock_client.upsert.assert_called_once()
    call_args = mock_client.upsert.call_args[1]
    assert call_args["collection_name"] == "polylogue_messages"
    assert len(call_args["points"]) == 1
    point = call_args["points"][0]
    assert point.payload["message_id"] == "m1"
    assert point.vector == [0.1, 0.2]

@patch("polylogue.index_qdrant.VectorStore")
def test_update_qdrant_for_conversations(mock_store_cls):
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    
    conn = MagicMock()
    # Mock execute result
    conn.execute.return_value.fetchall.return_value = [
        {"message_id": "m1", "conversation_id": "c1", "provider_name": "p1", "text": "content"}
    ]
    
    update_qdrant_for_conversations(["c1"], conn)
    
    mock_store.ensure_collection.assert_called_once()
    mock_store.upsert_messages.assert_called_once()
    msgs = mock_store.upsert_messages.call_args[0][0]
    assert msgs[0]["message_id"] == "m1"
