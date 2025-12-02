{ pkgs, extraPythonPackages ? [] }:

let
  py = pkgs.python3;
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
    tiktoken
    ijson
    qdrant-client
  ];

  pythonEnv = py.withPackages (ps: commonDeps ++ extraPythonPackages);
in
pkgs.mkShell {
  buildInputs =
    [ pythonEnv ]
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
    if [ -n "$BASH_VERSION" ]; then
      _polylogue_comp_file="$TMPDIR/polylogue-bash-completions"
      python3 polylogue.py completions --shell bash >"$_polylogue_comp_file" 2>/dev/null || true
      [ -f "$_polylogue_comp_file" ] && source "$_polylogue_comp_file"
    elif [ -n "$ZSH_VERSION" ]; then
      _polylogue_comp_file="$TMPDIR/polylogue-zsh-completions"
      python3 polylogue.py completions --shell zsh >"$_polylogue_comp_file" 2>/dev/null || true
      [ -f "$_polylogue_comp_file" ] && source "$_polylogue_comp_file"
    elif [ -n "$FISH_VERSION" ]; then
      _polylogue_comp_file="$TMPDIR/polylogue-fish-completions.fish"
      python3 polylogue.py completions --shell fish >"$_polylogue_comp_file" 2>/dev/null || true
      [ -f "$_polylogue_comp_file" ] && source "$_polylogue_comp_file"
    fi
  '';
}
