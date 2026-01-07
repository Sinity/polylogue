from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates

import polylogue
from polylogue.lib.repository import ConversationRepository
from polylogue.server.deps import get_repository

router = APIRouter()

# Locate templates relative to package
templates_dir = Path(polylogue.__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/")
def index(request: Request, repo: ConversationRepository = Depends(get_repository)):
    convs = repo.list(limit=50)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "convs": convs,
        },
    )


@router.get("/view/{conversation_id}")
def view_conversation(request: Request, conversation_id: str, repo: ConversationRepository = Depends(get_repository)):
    conv = repo.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return templates.TemplateResponse("modern.html", {"request": request, "conversation": conv})
