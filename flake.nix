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
        # Apply Python package overrides (fix dependency-injector marked as broken)
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              python313Packages = prev.python313Packages.overrideScope (
                pyfinal: pysuper: {
                  dependency-injector = pysuper.dependency-injector.overridePythonAttrs (old: {
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pyfinal.cython ];
                    doCheck = false;
                    meta = (old.meta or { }) // {
                      broken = false;
                    };
                  });
                }
              );
            })
          ];
        };
        python = pkgs.python313;

        # Build polylogue package
        polylogue = pkgs.python313Packages.buildPythonApplication {
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
            setuptools
            wheel
          ];

          dependencies = with pkgs.python313Packages; [
            google-auth-oauthlib
            google-api-python-client
            google-auth-httplib2
            httpx
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
            fastapi
            uvicorn
            orjson
            structlog
            pydantic
            pydantic-settings
            dependency-injector
            aiosqlite
            glom
          ];

          # Skip tests in build (run in checks instead)
          doCheck = false;
        };
      in
      {
        packages.default = polylogue;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python + uv for dependency management
            python
            uv

            # Development tools
            git
            ruff
            mypy

            # Runtime dependencies (CLI helpers)
            bat
            glow
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

            # Create venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv
            fi

            # Activate venv
            source .venv/bin/activate

            # Install dependencies if needed
            if [ ! -f .venv/.synced ]; then
              echo "Installing dependencies..."
              uv pip install -e ".[dev]"
              touch .venv/.synced
            fi

            echo "Polylogue development environment ready"
            echo "Run: polylogue --help"
          '';
        };

        # Simple check: run tests
        checks.default =
          pkgs.runCommand "polylogue-tests"
            {
              buildInputs = [
                python
                pkgs.uv
              ];
            }
            ''
              cp -r ${./.} source
              cd source
              export HOME=$TMPDIR
              ${pkgs.uv}/bin/uv venv
              source .venv/bin/activate
              ${pkgs.uv}/bin/uv pip install -e ".[dev]"
              pytest -q
              touch $out
            '';
      }
    )
    // {
      nixosModules = {
        polylogue = import ./nixos-modules/polylogue.nix { inherit self; };
        sync = import ./nixos-modules/sync.nix;
        default = self.nixosModules.polylogue;
      };
    };
}
