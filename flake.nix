{
  description = "Python dev shell with Google API + tools (single source of truth)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              re2 = prev.re2.overrideAttrs (old: {
                src = prev.fetchFromGitHub {
                  owner = "google";
                  repo = "re2";
                  rev = "ac82d4f628a2045d89964ae11c48403d3b091af1";
                  hash = "sha256-qRNV0O55L+r2rNSUJjU6nMqkPWXENZQvyr5riTU3e5o=";
                };
              });
            })
          ];
        };
      in {
        devShells.default = import ./nix/devshell.nix { inherit pkgs; };

        checks.default = pkgs.runCommand "aichat-to-md-pytest" {
          buildInputs = [
            (pkgs.python3.withPackages (ps: with ps; [
              google-api-python-client
              google-auth-oauthlib
              google-auth-httplib2
              pathvalidate
              aiohttp
              aiofiles
              google-generativeai
              rich
              pydantic
              python-frontmatter
              jinja2
              markdown-it-py
              pytest
            ]))
          ];
        } ''
          export PYTHONPATH=$PWD
          python -m pytest
          mkdir -p $out
        '';
      }
    );
}
