{
  description = "Python dev shell with Google API + tools (single source of truth)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      inherit (flake-utils.lib) eachDefaultSystem;

      re2Overlay = final: prev: {
        re2 = prev.re2.overrideAttrs (_: {
          src = prev.fetchFromGitHub {
            owner = "google";
            repo = "re2";
            rev = "ac82d4f628a2045d89964ae11c48403d3b091af1";
            hash = "sha256-qRNV0O55L+r2rNSUJjU6nMqkPWXENZQvyr5riTU3e5o=";
          };
        });
      };

    in
    eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ re2Overlay ];
        };

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
          qdrant-client
          ijson
        ];

        cliDeps = with pkgs; [ gum skim delta git ];
        cliBinPath = pkgs.lib.makeBinPath cliDeps;

        polylogueApp = pyPkgs.buildPythonApplication {
          pname = "polylogue";
          version = "0.1.0";
          pyproject = true;
          src = self;
          propagatedBuildInputs = commonDeps;
          nativeBuildInputs =
            (with pyPkgs; [ setuptools wheel ])
            ++ cliDeps
            ++ [ pkgs.makeWrapper ];
          nativeCheckInputs = (with pyPkgs; [ pytest ]) ++ cliDeps;
          checkPhase = ''
            export HOME=$TMPDIR
            export XDG_STATE_HOME=$TMPDIR
            export XDG_CACHE_HOME=$TMPDIR
            pytest
          '';
          postInstall = ''
            wrapProgram $out/bin/polylogue \
              --prefix PATH : ${cliBinPath}
          '';
        };

        defaultDevShell = import ./nix/devshell.nix {
          inherit pkgs;
          extraPythonPackages = with pyPkgs; [
            pytest
            pytest-cov
            coverage
            mypy
            types-requests
          ];
        };

        cliApp = {
          type = "app";
          program = "${polylogueApp}/bin/polylogue";
        };
      in {
        packages = {
          polylogue = polylogueApp;
          default = polylogueApp;
        };

        apps = {
          polylogue = cliApp;
          default = cliApp;
        };

        devShells = {
          default = defaultDevShell;
          ci = pkgs.mkShell {
            buildInputs = [ polylogueApp pkgs.git pkgs.which ];
            shellHook = ''
              export PATH=${polylogueApp}/bin:''${PATH}
              echo "Using packaged polylogue at ${polylogueApp}/bin/polylogue"
            '';
          };
        };

        checks.default = polylogueApp;
      }
    );
}
