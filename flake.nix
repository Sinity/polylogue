{
  description = "Python dev shell with Google API + tools (single source of truth)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      inherit (flake-utils.lib) eachDefaultSystem;

      re2Overlay = final: prev:
        let
          inherit (final) lib;
        in
        {
          re2 = prev.re2.overrideAttrs (old: {
            src = prev.fetchFromGitHub {
              owner = "google";
              repo = "re2";
              rev = "ac82d4f628a2045d89964ae11c48403d3b091af1";
              hash = "sha256-qRNV0O55L+r2rNSUJjU6nMqkPWXENZQvyr5riTU3e5o=";
            };
            postInstall = lib.concatStringsSep "\n" [
              (old.postInstall or "")
              ''
                patch_re2_config() {
                  local file="$1"
                  if [[ ! -f "$file" ]]; then
                    return
                  fi

                  tmp="$(mktemp)"
                  awk '
                    /^set_and_check\(re2_INCLUDE_DIR/ {next}
                    /^include\(CMakeFindDependencyMacro\)$/ {
                      print
                      print ""
                      print "set_and_check(re2_INCLUDE_DIR ''${PACKAGE_PREFIX_DIR}/include)"
                      next
                    }
                    {print}
                  ' "$file" > "$tmp"
                  mv "$tmp" "$file"
                }

                patch_re2_config "$out/lib/cmake/re2/re2Config.cmake"
                if [[ -n "$dev" ]]; then
                  patch_re2_config "$dev/lib/cmake/re2/re2Config.cmake"
                fi
              ''
            ];
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

          cliDeps = with pkgs; [ git bat glow skim ];
          cliBinPath = pkgs.lib.makeBinPath cliDeps;

          polylogueApp = pyPkgs.buildPythonApplication {
            pname = "polylogue";
            version = "0.1.0";
            pyproject = true;
            doCheck = false;
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

              generate_completion() {
                local shell="$1"
                local target="$2"
                if ! $out/bin/polylogue completions --shell "$shell" > "$target"; then
                  echo "warning: skipping polylogue completions for $shell" >&2
                  rm -f "$target"
                fi
              }

              generate_completion bash $out/share/bash-completion/completions/polylogue
              generate_completion zsh $out/share/zsh/site-functions/_polylogue
              generate_completion fish $out/share/fish/vendor_completions.d/polylogue.fish
            '';
          };
          polylogueChecks = polylogueApp.overridePythonAttrs (_: { doCheck = true; });

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

          checks.default = polylogueChecks;
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
