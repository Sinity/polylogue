{
  description = "Python dev shell with Google API + tools (single source of truth)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      polylogueModule = import ./nix/modules/polylogue.nix { inherit self; };
    in
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
        python = pkgs.python3;
        pyPkgs = pkgs.python3Packages;
        baseDeps =
          with pyPkgs;
          [
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
            qdrant-client
          ];
        polylogueApp = pyPkgs.buildPythonApplication {
          pname = "polylogue";
          version = "0.1.0";
          pyproject = true;
          src = self;
          propagatedBuildInputs = baseDeps;
          nativeBuildInputs = with pyPkgs; [
            setuptools
            wheel
          ];
          nativeCheckInputs =
            (with pyPkgs; [ pytest ])
            ++ (with pkgs; [ delta gum git ]);
          checkPhase = "pytest";
        };
        cliApp = {
          type = "app";
          program = "${polylogueApp}/bin/polylogue";
        };
      in {
        packages = {
          default = polylogueApp;
          polylogue = polylogueApp;
        };

        apps = {
          default = cliApp;
          polylogue = cliApp;
        };

        devShells = {
          default = import ./nix/devshell.nix {
            inherit pkgs;
            extraPythonPackages = with pkgs.python3Packages; [
              pytest
              pytest-cov
              coverage
              mypy
              types-requests
            ];
          };
          ci = pkgs.mkShell {
            buildInputs = [ polylogueApp pkgs.git pkgs.which ];
            shellHook = ''
              export PATH=${polylogueApp}/bin:''${PATH}
              echo "Using packaged polylogue at ${polylogueApp}/bin/polylogue"
            '';
          };
        };

        checks = {
          default = polylogueApp;
        };
      }
    ) // {
      nixosModules = {
        default = polylogueModule;
        polylogue = polylogueModule;
      };
    };
}
