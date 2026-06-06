{
  description = "Polylogue - AI session archive";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };
      python = pkgs.python313;

      # Script body lives in nix/devtools-wrapper.sh so it can be unit-tested
      # directly (see tests/unit/devtools/test_cli_wrapper.py).
      devtoolsCli = pkgs.writeShellScriptBin "devtools" (builtins.readFile ./nix/devtools-wrapper.sh);

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
          license = pkgs.lib.licenses.mit;
          homepage = "https://github.com/Sinity/polylogue";
          platforms = pkgs.lib.platforms.linux;
        };
      };

      # Python environment with polylogue pre-installed (for scripting/notebooks).
      polylogueApiPython = python.withPackages (_: [ polylogue ]);

      # Sanitized api-python: the python binary is wrapped with the same env
      # sanitization as the CLI binaries so downstream consumers (sinnix) don't
      # need their own wrapper.
      polylogueApiPythonWrapped = pkgs.runCommand "polylogue-api-python-wrapped"
        {
          buildInputs = [ pkgs.makeWrapper ];
        }
        ''
          mkdir -p "$out/bin"
          for f in ${polylogueApiPython}/bin/*; do
            name=$(basename "$f")
            case "$name" in
              python|python3|python3.*)
                makeWrapper "$f" "$out/bin/$name" \
                  --unset PYTHONPATH \
                  --unset PYTHONHOME \
                  --unset PYTHONBREAKPOINT \
                  --unset PYTHONUSERBASE \
                  --unset VIRTUAL_ENV
                ;;
              *)
                ln -s "$f" "$out/bin/$name"
                ;;
            esac
          done
          for d in lib include share; do
            if [ -d "${polylogueApiPython}/$d" ]; then
              ln -s "${polylogueApiPython}/$d" "$out/$d"
            fi
          done
        '';
    in
    {
      packages.${system} = {
        inherit polylogue;
        default = polylogue;
        api-python = polylogueApiPythonWrapped;
        api-python-raw = polylogueApiPython;
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python
          uv
          devtoolsCli
          git
          ruff
          mypy
          py-spy
          python.pkgs.pyinstrument
          bat
          glow
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

          # One-time cleanup of legacy repo-root cache dirs (migrated under .cache/).
          legacy_stamp=".cache/.legacy-caches-migrated"
          if [ ! -f "$legacy_stamp" ]; then
            for legacy_cache_root in __pycache__ .pytest_cache .hypothesis .mypy_cache .ruff_cache .benchmarks; do
              if [ -e "$legacy_cache_root" ]; then
                rm -rf "$legacy_cache_root"
              fi
            done
            touch "$legacy_stamp"
          fi

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
            echo "devshell: creating virtual environment" >&2
            uv venv
          fi

          # Activate venv
          source .venv/bin/activate

          # Sync dependencies when pyproject.toml, uv.lock, or Python version change.
          sync_fingerprint_file=".venv/.uv-sync-fingerprint"
          sync_fingerprint="$(
            cat pyproject.toml uv.lock 2>/dev/null
            python --version 2>&1
          )"
          sync_fingerprint="$(printf '%s' "$sync_fingerprint" | sha256sum | cut -d' ' -f1)"
          current_fingerprint=""
          if [ -f "$sync_fingerprint_file" ]; then
            current_fingerprint="$(cat "$sync_fingerprint_file")"
          fi

          if [ "$sync_fingerprint" != "$current_fingerprint" ]; then
            echo "devshell: syncing Python dependencies (fingerprint changed)" >&2
            uv sync --extra dev --frozen --quiet
            printf '%s' "$sync_fingerprint" > "$sync_fingerprint_file"
          fi

          # Render AGENTS.md when CLAUDE.md or its @include targets change.
          # Uses content hashing (not mtime) to avoid spurious rebuilds on
          # touch / rsync / branch switches without content change.
          agents_hash_file=".cache/.last-agents-render-hash"
          agents_hash="$(
            cat CLAUDE.md CONTRIBUTING.md TESTING.md \
              docs/architecture.md docs/internals.md docs/devtools.md \
              2>/dev/null | sha256sum | cut -d' ' -f1
          )"
          agents_last_hash=""
          if [ -f "$agents_hash_file" ]; then
            agents_last_hash="$(cat "$agents_hash_file")"
          fi

          if [ "$agents_hash" != "$agents_last_hash" ]; then
            echo "devshell: regenerating AGENTS.md (sources changed)" >&2
            devtools render-agents >/dev/null
            printf '%s' "$agents_hash" > "$agents_hash_file"
          fi

          if [[ $- == *i* ]]; then
            devtools status --stderr || true
            export POLYLOGUE_MOTD_RENDERED=1
          fi
        '';
      };

      checks.${system} = {
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

        format = pkgs.runCommand "polylogue-format"
          {
            nativeBuildInputs = [
              pkgs.ruff
            ];
          }
          ''
            export RUFF_CACHE_DIR=$TMPDIR/.ruff-cache
            cd ${self}
            ruff format --check polylogue/ tests/ devtools/
            touch $out
          '';

        lint = pkgs.runCommand "polylogue-lint"
          {
            nativeBuildInputs = [
              pkgs.ruff
            ];
          }
          ''
            export RUFF_CACHE_DIR=$TMPDIR/.ruff-cache
            cd ${self}
            ruff check polylogue/ tests/ devtools/
            touch $out
          '';
      };

      formatter.${system} = pkgs.nixfmt;

      apps.${system} = {
        polylogue = {
          type = "app";
          program = "${polylogue}/bin/polylogue";
        };
        polylogued = {
          type = "app";
          program = "${polylogue}/bin/polylogued";
        };
        polylogue-mcp = {
          type = "app";
          program = "${polylogue}/bin/polylogue-mcp";
        };
        default = {
          type = "app";
          program = "${polylogue}/bin/polylogue";
        };
      };

      nixosModules.default = import ./nix/module.nix;
      nixosModules.polylogue = import ./nix/module.nix;
      homeManagerModules.default = import ./nix/hm-module.nix;
      homeManagerModules.polylogue = import ./nix/hm-module.nix;
    };
}
