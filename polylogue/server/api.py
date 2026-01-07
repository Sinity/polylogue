from fastapi import APIRouter, Depends, HTTPException, Query

from polylogue.lib.models import Conversation
from polylogue.lib.repository import ConversationRepository
from polylogue.server.deps import get_repository

router = APIRouter()


@router.get("/conversations", response_model=list[Conversation])
def list_conversations(limit: int = 50, offset: int = 0, repo: ConversationRepository = Depends(get_repository)):
    return repo.list(limit=limit, offset=offset)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str, repo: ConversationRepository = Depends(get_repository)):
    conv = repo.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/search", response_model=list[Conversation])
def search_conversations(q: str = Query(..., min_length=3), repo: ConversationRepository = Depends(get_repository)):
    return repo.search(q)
