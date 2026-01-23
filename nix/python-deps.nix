{ pkgs }:
let
  pyPkgs = pkgs.python3Packages;
  commonDeps = with pyPkgs; [
    google-auth-oauthlib
    google-api-python-client
    google-auth-httplib2
    httpx
    rich
    pydantic
    jinja2
    markdown-it-py
    ijson
    qdrant-client
    questionary
    pygments
    tenacity
    click
    dateparser
    fastapi
    uvicorn
    structlog
    orjson
    dependency-injector
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
