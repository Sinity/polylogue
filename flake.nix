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
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
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
            BUILD_COMMIT = "${self.shortRev or self.dirtyShortRev or "unknown"}"
            BUILD_DIRTY = ${if self ? dirtyShortRev then "True" else "False"}
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
            aiosqlite
            mcp
            pyyaml
            watchfiles
          ];

          # Skip tests in build (run in checks instead)
          doCheck = false;
          pythonImportsCheck = [
            "polylogue"
          ];
          dontCheckRuntimeDeps = true;

          postFixup = ''
            for program in polylogue polylogued polylogue-mcp; do
              wrapProgram "$out/bin/$program" \
                --unset PYTHONPATH \
                --unset PYTHONHOME \
                --unset PYTHONBREAKPOINT \
                --unset PYTHONUSERBASE \
                --unset VIRTUAL_ENV
            done
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

            # One-time cleanup of legacy repo-root cache dirs migrated under .cache/.
            for legacy_cache_root in __pycache__ .pytest_cache .hypothesis .mypy_cache .ruff_cache .benchmarks; do
              if [ -e "$legacy_cache_root" ]; then
                rm -rf "$legacy_cache_root"
              fi
            done

            # Install repo git hooks — skip if already set correctly.
            current_hooks_path=$(git config --local core.hooksPath 2>/dev/null || true)
            if [ "$current_hooks_path" != ".githooks" ]; then
              git config --local core.hooksPath .githooks 2>/dev/null || true
            fi

            # Clean stale __pycache__ dirs under source trees — skip if stamp
            # is fresh (sources haven't changed since last cleanup).
            pyc_stamp=".cache/.last-pyc-cleanup"
            pyc_should_clean=1
            if [ -f "$pyc_stamp" ]; then
              last_clean=$(stat -c %Y "$pyc_stamp" 2>/dev/null || echo 0)
              newest_src=$(find polylogue tests devtools -name '*.py' -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)
              if [ -n "$newest_src" ] && [ "$last_clean" -ge "$newest_src" ]; then
                pyc_should_clean=0
              fi
            fi
            if [ "$pyc_should_clean" -eq 1 ]; then
              find polylogue tests devtools -type d -name __pycache__ -prune -exec rm -r {} + 2>/dev/null || true
              touch "$pyc_stamp"
            fi

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

            # Render AGENTS.md when CLAUDE.md or its inclusions change.
            agents_stamp=".cache/.last-agents-render"
            if [ -f CLAUDE.md ]; then
              agents_src_mtime=$(stat -c %Y CLAUDE.md 2>/dev/null || echo 0)
              agents_last_mtime=$(stat -c %Y "$agents_stamp" 2>/dev/null || echo 0)
              if [ "$agents_src_mtime" -gt "$agents_last_mtime" ]; then
                devtools render-agents >/dev/null
                touch "$agents_stamp"
              fi
            fi

            if [[ $- == *i* ]]; then
              devtools status --stderr || true
              export POLYLOGUE_MOTD_RENDERED=1
            fi
          '';
        };

        # Smoke check the packaged CLI inside a Nix sandbox. Full test coverage
        # lives in GitHub Actions and local dev workflows.
        checks = {
          default = pkgs.runCommand "polylogue-smoke"
            {
              nativeBuildInputs = [
                polylogue
              ];
            }
            ''
              export HOME=$TMPDIR
              polylogue --help >/dev/null
              polylogued --help >/dev/null
              polylogue-mcp --help >/dev/null
              touch $out
            '';

          # Gate: code must be formatted.
          format = pkgs.runCommand "polylogue-format"
            {
              nativeBuildInputs = [
                pkgs.ruff
              ];
            }
            ''
              cd ${self}
              ruff format --check polylogue/ tests/ devtools/
              touch $out
            '';

          # Gate: code must pass lint.
          lint = pkgs.runCommand "polylogue-lint"
            {
              nativeBuildInputs = [
                pkgs.ruff
              ];
            }
            ''
              cd ${self}
              ruff check polylogue/ tests/ devtools/
              touch $out
            '';
        };

        formatter = pkgs.nixfmt;
      }
    ) // {
      nixosModules.default = import ./nix/module.nix;
    };
}
