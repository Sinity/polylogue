{ pkgs, extraPythonPackages ? [] }:

let
  py = pkgs.python3;
  deps = import ./python-deps.nix { inherit pkgs; };
  pythonEnv = py.withPackages (ps: deps.commonDeps ++ extraPythonPackages);
in
pkgs.mkShell {
  buildInputs =
    [ pythonEnv ]
    ++ [
      # CLI helpers used by Polylogue
      pkgs.fd
      pkgs.ripgrep
      pkgs.jq
      pkgs.bat
      pkgs.skim
      pkgs.glow
      pkgs.tesseract
    ];
  shellHook = ''
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    export TESSDATA_PREFIX="${pkgs.tesseract}/share/tessdata"
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
