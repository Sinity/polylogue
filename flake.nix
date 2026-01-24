{
  description = "Polylogue - AI conversation archive";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;
      in
      {
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
        checks.default = pkgs.runCommand "polylogue-tests" {
          buildInputs = [ python pkgs.uv ];
        } ''
          cp -r ${./.} source
          cd source
          export HOME=$TMPDIR
          ${pkgs.uv}/bin/uv venv
          source .venv/bin/activate
          ${pkgs.uv}/bin/uv pip install -e ".[dev]"
          pytest -q --ignore=tests/test_qdrant.py
          touch $out
        '';
      }
    ) // {
      # NixOS module for polylogue-sync service
      nixosModules.default = import ./nixos-modules/polylogue-sync.nix;
    };
}
