{ pkgs }:
let
  pyPkgs = pkgs.python3Packages;
  commonDeps = with pyPkgs; [
    google-auth-oauthlib
    requests
    pathvalidate
    aiohttp
    aiofiles
    rich
    pydantic
    python-frontmatter
    jinja2
    markdown-it-py
    pyperclip
    watchfiles
    ijson
    tiktoken
    qdrant-client
  ];
  devDeps = with pyPkgs; [
    pytest
    pytest-cov
    coverage
    mypy
    types-requests
  ];
in {
  inherit commonDeps devDeps;
}
