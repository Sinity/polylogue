{
  description = "Polylogue";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      # The package build proves a clean, locked build. Tests run through
      # `cargo run -p xtask -- check`, which reuses incremental builds.
      polylogue = pkgs.rustPlatform.buildRustPackage {
        pname = "polylogue";
        version = "0.1.0";
        src = self;
        cargoLock.lockFile = ./Cargo.lock;
        nativeBuildInputs = [ pkgs.pkg-config ];
        buildInputs = [ pkgs.zstd ];
        cargoBuildFlags = [ "-p" "polylogue" ];
        doCheck = false;
      };
    in {
      packages.${system} = { default = polylogue; inherit polylogue; };
      checks.${system}.polylogue = polylogue;
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.cargo pkgs.cargo-nextest pkgs.rustc pkgs.rustfmt pkgs.clippy pkgs.gcc pkgs.pkg-config pkgs.zstd ];
      };
    };
}
