{ pkgs }:

pkgs.mkShell {
  buildInputs = [
    # Python + libraries
    pkgs.python3
    pkgs.python3Packages.google-api-python-client
    pkgs.python3Packages.google-auth-oauthlib
    pkgs.python3Packages.google-auth-httplib2
    pkgs.python3Packages.pathvalidate
    pkgs.python3Packages.aiohttp
    pkgs.python3Packages.aiofiles
    pkgs.python3Packages.google-generativeai
    pkgs.python3Packages.rich
    pkgs.python3Packages.pydantic
    pkgs.python3Packages.python-frontmatter
    pkgs.python3Packages.jinja2
    pkgs.python3Packages.markdown-it-py

    # CLI helpers used by Polylogue
    pkgs.skim
    pkgs.gum
    pkgs.bat
    pkgs.delta
    pkgs.fd
    pkgs.ripgrep
    pkgs.glow
    pkgs.jq
  ];
  shellHook = ''
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    export SKIM_DEFAULT_COMMAND="rg --files"
    export SKIM_DEFAULT_OPTIONS="--ansi"
    export GUM_STYLE_FOREGROUND="#7fdbca"
    export GUM_CONFIRM_PROMPT_BORDER_FOREGROUND="#ff9f1c"
  '';
}
