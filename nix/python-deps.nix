{ pkgs }:
let
  pyPkgs = pkgs.python3Packages;
  commonDeps = with pyPkgs; [
    google-auth-oauthlib
    google-api-python-client
    google-auth-httplib2
    httpx
    pathvalidate
    aiofiles
    rich
    pydantic
    pydantic-settings
    python-frontmatter
    jinja2
    markdown-it-py
    watchfiles
    ijson
    tiktoken
    qdrant-client
    pypdf
    questionary
    pygments
    tenacity
    click
    pillow
    pytesseract
    fastapi
    uvicorn
    structlog
    orjson
  ];
  devDeps = with pyPkgs; [
    pytest
    pytest-cov
    coverage
    mypy
    pexpect
  ];
in {
  inherit commonDeps devDeps;
}
