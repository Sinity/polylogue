{
  description = "Polylogue isolated Rust foundation";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      foundation = pkgs.rustPlatform.buildRustPackage {
        pname = "polylogue";
        version = "0.1.0";
        src = self;
        cargoLock.lockFile = ./Cargo.lock;
        nativeBuildInputs = [ pkgs.pkg-config ];
        buildInputs = [ pkgs.zstd ];
        doCheck = true;
      };
    in {
      packages.${system} = { default = foundation; polylogue = foundation; };
      checks.${system}.foundation = foundation;
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.cargo pkgs.rustc pkgs.rustfmt pkgs.clippy pkgs.gcc pkgs.pkg-config pkgs.zstd pkgs.python3 ];
      };
    };
}
