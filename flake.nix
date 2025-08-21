{
  description = "Python dev shell with Google API + fixed re2";

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
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python3
            pkgs.python3Packages.google-api-python-client
            pkgs.python3Packages.google-auth-oauthlib
            pkgs.python3Packages.google-auth-httplib2
            pkgs.python3Packages.aiohttp
            pkgs.python3Packages.tqdm
            pkgs.python3Packages.pathvalidate
            pkgs.python3Packages.aiofiles
          ];
        };
      }
    );
}

