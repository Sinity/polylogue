{
  description = "Polylogue - AI conversation archive";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        python = pkgs.python313;
        devtoolsCli = pkgs.writeShellScriptBin "devtools" ''
          if [ -z "''${POLYLOGUE_REPO_ROOT:-}" ]; then
            echo "devtools: POLYLOGUE_REPO_ROOT is not set; enter the project devshell first" >&2
            exit 1
          fi
          exec python "$POLYLOGUE_REPO_ROOT/devtools/__main__.py" "$@"
        '';

        # Build polylogue package as an importable library that also exposes the
        # project console script from pyproject entry points.
        polylogue = pkgs.python313Packages.buildPythonPackage {
          pname = "polylogue";
          version = "0.1.0";
          pyproject = true;
          src = ./.;

          postPatch = ''
            cat > polylogue/_build_info.py << BUILDEOF
            BUILD_COMMIT = "${self.rev or self.dirtyRev or "unknown"}"
            BUILD_DIRTY = ${if self ? dirtyRev then "True" else "False"}
            BUILDEOF
          '';

          build-system = with pkgs.python313Packages; [
            hatchling
          ];

          nativeBuildInputs = [
            pkgs.makeWrapper
          ];

          dependencies = with pkgs.python313Packages; [
            google-auth-oauthlib
            google-api-python-client
            google-auth-httplib2
            httpx
            h2
            rich
            textual
            jinja2
            markdown-it-py
            pygments
            ijson
            sqlite-vec
            questionary
            click
            tenacity
            dateparser
            orjson
            structlog
            pydantic
            pydantic-settings
            aiosqlite
            glom
            mcp
            pyyaml
          ];

          # Skip tests in build (run in checks instead)
          doCheck = false;
          pythonImportsCheck = [
            "polylogue"
            "devtools"
          ];
          dontCheckRuntimeDeps = true;

          postFixup = ''
            wrapProgram "$out/bin/polylogue" \
              --unset PYTHONPATH \
              --unset PYTHONHOME \
              --unset PYTHONBREAKPOINT \
              --unset PYTHONUSERBASE \
              --unset VIRTUAL_ENV
          '';

          meta = {
            description = "Polylogue archive Python package and CLI";
            mainProgram = "polylogue";
          };
        };
        polylogueApiPython = python.withPackages (_: [ polylogue ]);
      in
      {
        packages.polylogue = polylogue;
        packages.default = polylogue;
        packages.api-python = polylogueApiPython;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python + uv for dependency management
            python
            uv
            devtoolsCli

            # Development tools
            git
            ruff
            mypy

            # Profiling
            py-spy
            python.pkgs.pyinstrument

            # Runtime dependencies (CLI helpers)
            bat
            glow

            # Demo screencast recording
            vhs
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            export HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.cache/hypothesis"
            export PYTHONDONTWRITEBYTECODE=1
            export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
            export POLYLOGUE_REPO_ROOT="$PWD"
            mkdir -p .cache .local "$PYTHONPYCACHEPREFIX"

            if [ -L result ]; then
              rm result
            fi

            for legacy_cache_root in __pycache__ .pytest_cache .hypothesis .mypy_cache .ruff_cache .benchmarks; do
              if [ -e "$legacy_cache_root" ]; then
                rm -rf "$legacy_cache_root"
              fi
            done

            # Install repo git hooks (format/lint on commit, verify on push).
            git config --local core.hooksPath .githooks 2>/dev/null || true
            find polylogue tests devtools -type d -name __pycache__ -prune -exec rm -r {} + 2>/dev/null || true

            # Create venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..." >&2
              uv venv
            fi

            # Activate venv
            source .venv/bin/activate

            sync_fingerprint_file=".venv/.uv-sync-fingerprint"
            sync_fingerprint="$(
              cat pyproject.toml uv.lock 2>/dev/null | sha256sum | cut -d' ' -f1
            )"
            current_fingerprint=""
            if [ -f "$sync_fingerprint_file" ]; then
              current_fingerprint="$(cat "$sync_fingerprint_file")"
            fi

            # Sync dependencies whenever pyproject or uv.lock changes.
            if [ "$sync_fingerprint" != "$current_fingerprint" ]; then
              if [[ $- == *i* ]]; then
                echo "devshell: syncing Python dependencies" >&2
              fi
              uv sync --extra dev --frozen --quiet
              printf '%s' "$sync_fingerprint" > "$sync_fingerprint_file"
            fi

            if [ -f CLAUDE.md ]; then
              devtools render-agents >/dev/null
            fi

            if [[ $- == *i* ]]; then
              devtools status --stderr || true
              export POLYLOGUE_MOTD_RENDERED=1
            fi
          '';
        };

        # Smoke check the packaged CLI inside a Nix sandbox. Full test coverage
        # lives in GitHub Actions and local dev workflows.
        checks.default =
          pkgs.runCommand "polylogue-smoke"
            {
              nativeBuildInputs = [
                polylogue
              ];
            }
            ''
              export HOME=$TMPDIR
              polylogue --help >/dev/null
              touch $out
            '';
      }
    );
}
