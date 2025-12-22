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

      perSystem = eachDefaultSystem (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ re2Overlay ];
          };

          pyPkgs = pkgs.python3Packages;
          deps = import ./nix/python-deps.nix { inherit pkgs; };

          cliDeps = with pkgs; [ git ];
          cliBinPath = pkgs.lib.makeBinPath cliDeps;

          polylogueApp = pyPkgs.buildPythonApplication {
            pname = "polylogue";
            version = "0.1.0";
            pyproject = true;
            src = self;
            propagatedBuildInputs = deps.commonDeps;
            nativeBuildInputs =
              (with pyPkgs; [ setuptools wheel ])
              ++ cliDeps
              ++ [ pkgs.makeWrapper ];
            nativeCheckInputs = deps.devDeps ++ cliDeps;
            checkPhase = ''
              export HOME=$TMPDIR
              export XDG_STATE_HOME=$TMPDIR
              export XDG_CACHE_HOME=$TMPDIR
              pytest
            '';
            postInstall = ''
              wrapProgram $out/bin/polylogue \
                --prefix PATH : ${cliBinPath}

              mkdir -p $out/share/bash-completion/completions
              mkdir -p $out/share/zsh/site-functions
              mkdir -p $out/share/fish/vendor_completions.d

              $out/bin/polylogue completions --shell bash > $out/share/bash-completion/completions/polylogue
              $out/bin/polylogue completions --shell zsh > $out/share/zsh/site-functions/_polylogue
              $out/bin/polylogue completions --shell fish > $out/share/fish/vendor_completions.d/polylogue.fish
            '';
          };

          defaultDevShell = import ./nix/devshell.nix {
            inherit pkgs;
            extraPythonPackages = deps.devDeps;
          };

          cliApp = {
            type = "app";
            program = "${polylogueApp}/bin/polylogue";
            meta = {
              description = "Polylogue CLI";
            };
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
    in
    perSystem // {
      nixosModules = {
        polylogue = import ./nix/modules/polylogue.nix { self = self; };
        default = self.nixosModules.polylogue;
      };
    };
}
