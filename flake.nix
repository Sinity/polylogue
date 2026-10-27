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
        # The shared crates are Git dependencies; Nix fetches them at the
        # locked commit.
        cargoLock = { lockFile = ./Cargo.lock; allowBuiltinFetchGit = true; };
        nativeBuildInputs = [ pkgs.pkg-config ];
        buildInputs = [ pkgs.zstd ];
        cargoBuildFlags = [ "-p" "polylogue" ];
        # Primary mode is chosen by `sinex.toml` at run time, so the packaged
        # binary carries it.
        buildFeatures = [ "sinex" ];
        doCheck = false;
      };
    in {
      packages.${system} = { default = polylogue; inherit polylogue; };
      checks.${system}.polylogue = polylogue;
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.cargo pkgs.cargo-nextest pkgs.rustc pkgs.rustfmt pkgs.clippy pkgs.nodejs_22 pkgs.gcc pkgs.pkg-config pkgs.zstd pkgs.nats-server pkgs.mold pkgs.sqlite ];
        # mold links the test binaries several times faster than the default
        # linker: on a cold test build the workspace's test binaries took
        # 169 s of linking with ld and 37 s with mold.
        CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS = "-C link-arg=-fuse-ld=mold";
        shellHook = ''
          # Optional host placement; an explicitly set target directory is kept.
          if [ -z "''${CARGO_TARGET_DIR:-}" ] && command -v sinnix-cargo-target >/dev/null 2>&1; then
            if cargo_target="$(sinnix-cargo-target)"; then
              export CARGO_TARGET_DIR="$cargo_target"
            fi
          fi
          export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-$PWD/target}"
        '';
      };
    };
}
