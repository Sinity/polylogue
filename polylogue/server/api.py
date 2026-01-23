from fastapi import APIRouter, Depends, HTTPException, Query

from polylogue.storage.db import DatabaseError
from polylogue.lib.models import Conversation
from polylogue.lib.repository import ConversationRepository
from polylogue.server.deps import get_repository

router = APIRouter()


@router.get("/conversations", response_model=list[Conversation])
def list_conversations(limit: int = 50, offset: int = 0, repo: ConversationRepository = Depends(get_repository)) -> list[Conversation]:
    return repo.list(limit=limit, offset=offset)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str, repo: ConversationRepository = Depends(get_repository)) -> Conversation:
    conv = repo.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/search", response_model=list[Conversation])
def search_conversations(q: str = Query(..., min_length=3), repo: ConversationRepository = Depends(get_repository)) -> list[Conversation]:
    try:
        return repo.search(q)
    except (RuntimeError, DatabaseError) as exc:
        message = str(exc)
        if "index not built" in message.lower() or "search index" in message.lower():
            raise HTTPException(
                status_code=503,
                detail="Search index not built. Run `polylogue run` or `polylogue index` first.",
            ) from exc
        raise HTTPException(status_code=500, detail=message) from exc
