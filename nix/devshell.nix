{ pkgs, extraPythonPackages ? [] }:

let
  py = pkgs.python3;
  pyPkgs = pkgs.python3Packages;
  pythonLibs = with pyPkgs; [
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
  ] ++ extraPythonPackages;
in
pkgs.mkShell {
  buildInputs =
    [ py ]
    ++ pythonLibs
    ++ [
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
