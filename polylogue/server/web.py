from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

import polylogue
from polylogue.lib.repository import ConversationRepository
from polylogue.server.deps import get_repository

router = APIRouter()

# Locate templates relative to package
templates_dir = Path(polylogue.__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
_markdown = MarkdownIt("commonmark", {"linkify": True, "html": False}).enable("table")


def _render_markdown(text: str | None) -> str:
    if not text:
        return ""
    return _markdown.render(text)


templates.env.filters["render_markdown"] = _render_markdown


@router.get("/")
def index(request: Request, repo: ConversationRepository = Depends(get_repository)):
    convs = repo.list(limit=50)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "convs": convs,
            "conversation": convs[0] if convs else None,
        },
    )


@router.get("/view/{conversation_id}")
def view_conversation(request: Request, conversation_id: str, repo: ConversationRepository = Depends(get_repository)):
    conv = repo.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return templates.TemplateResponse(request, "modern.html", {"request": request, "conversation": conv})
