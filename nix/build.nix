{
  pkgs,
  crane,
  toolchain,
  source,
  revision,
}:
let
  lib = pkgs.lib;
  craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;
  workspace = (builtins.fromTOML (builtins.readFile ../Cargo.toml)).workspace.package;
  common = {
    src = source;
    pname = "polylogue";
    version = workspace.version;
    CARGO_PROFILE = "release";
    nativeBuildInputs = [ pkgs.pkg-config ];
    buildInputs = [ pkgs.zstd ];
    strictDeps = true;
  };
  # Build identity belongs to the application, not the dependency cache.
  cargoArtifacts = craneLib.buildDepsOnly (
    common
    // {
      src = craneLib.cleanCargoSource source;
      cargoExtraArgs = "--locked --workspace --all-features";
    }
  );
  application = common // {
    inherit cargoArtifacts;
    POLYLOGUE_REVISION = revision;
    doInstallCargoArtifacts = false;
  };
  polylogue = craneLib.buildPackage (
    application
    // {
      cargoExtraArgs = "--locked -p polylogue --features polylogue/sinex";
      doCheck = false;
      nativeBuildInputs = common.nativeBuildInputs ++ [ pkgs.makeBinaryWrapper ];
      postInstall = ''
        wrapProgram "$out/bin/polylogue" --inherit-argv0 --prefix PATH : ${lib.makeBinPath [ pkgs.git ]}
      '';
      meta.mainProgram = "polylogue";
    }
  );
  extensionVersion = (builtins.fromJSON (builtins.readFile ../extension/package.json)).version;
  extension =
    pkgs.runCommand "polylogue-extension-${extensionVersion}"
      {
        nativeBuildInputs = [
          pkgs.nodejs_22
          pkgs.zip
          pkgs.unzip
          pkgs.jq
        ];
      }
      ''
            cp -R ${../extension} extension
            chmod -R u+w extension
            node extension/scripts/validate-manifest.mjs
        node extension/scripts/build.mjs --no-source-sync --out "$out"
            mkdir -p "$out/share/polylogue/extension"
            chrome=$(jq -r '.artifacts.chrome' "$out/build-manifest.json")
            unzip -q "$out/$chrome" -d "$out/share/polylogue/extension"
      '';
  checks = {
    settings = import ./checks/settings.nix { inherit pkgs; };
    # Retain the existing package check name.
    inherit polylogue;
    fmt = craneLib.cargoFmt {
      inherit (common) src pname version;
      cargoExtraArgs = "--all";
    };
    clippy = craneLib.cargoClippy (
      application
      // {
        cargoExtraArgs = "--locked --workspace --features polylogue/sinex";
        cargoClippyExtraArgs = "--all-targets -- -D warnings";
      }
    );
    clippy-standalone = craneLib.cargoClippy (
      application
      // {
        cargoExtraArgs = "--locked -p polylogue";
        cargoClippyExtraArgs = "--all-targets -- -D warnings";
      }
    );
    nextest = craneLib.cargoNextest (
      application
      // {
        cargoExtraArgs = "--locked --workspace --exclude polylogue-sinex --features polylogue-engine/faults,polylogue-store/faults";
        # Shell completion stays offline; the remaining real-process tests need
        # an OS identity. Native checks retain every e2e; the VM covers transports.
        cargoNextestExtraArgs = "-E 'not (package(polylogue) and binary(e2e) and not test(/^completion::/))'";
        nativeBuildInputs = common.nativeBuildInputs ++ [
          pkgs.git
          pkgs.python3
          pkgs.dbus
          pkgs.nodejs_22
          pkgs.sqlite
        ];
        SHELL = pkgs.runtimeShell;
        TZDIR = "${pkgs.tzdata}/share/zoneinfo";
        preCheck = ''
          export XDG_STATE_HOME="$TMPDIR/state"
        '';
      }
    );
    boundaries = craneLib.mkCargoDerivation (
      application
      // {
        pname = "polylogue-boundaries";
        nativeBuildInputs = common.nativeBuildInputs ++ [ pkgs.git ];
        preBuild = ''
          export XDG_STATE_HOME="$TMPDIR/state"
        '';
        buildPhaseCargoCommand = "cargoWithProfile run --locked -p xtask -- boundaries";
      }
    );
    extension = extension;
    nix-fmt =
      pkgs.runCommand "polylogue-nix-format"
        {
          nativeBuildInputs = [ pkgs.nixfmt ];
        }
        ''
          find ${
            lib.fileset.toSource {
              root = ../.;
              fileset = lib.fileset.fileFilter (file: file.hasExt "nix") ../.;
            }
          } -name '*.nix' -print0 | xargs -0 -r nixfmt --check
          touch "$out"
        '';
    behavior = import ./checks/behavior.nix { inherit pkgs polylogue revision; };
  };
in
{
  packages = {
    inherit polylogue;
    polylogue-extension = extension;
    default = polylogue;
  };
  inherit checks;
  devShell = craneLib.devShell {
    # Entering the shell needs check tools, not a VM image or its driver.
    checks = lib.removeAttrs checks [ "behavior" ];
    packages = [
      toolchain
      pkgs.git
      pkgs.cargo-nextest
      pkgs.cargo-llvm-cov
      pkgs.cargo-fuzz
      pkgs.cargo-deny
      pkgs.nodejs_22
      pkgs.python3
      pkgs.gcc
      pkgs.pkg-config
      pkgs.zstd
      pkgs.nats-server
      pkgs.dbus
      pkgs.sqlite
      pkgs.jq
      pkgs.mold
    ];
    RUST_SRC_PATH = "${toolchain}/lib/rustlib/src/rust/library";
    LLVM_COV = "${toolchain}/lib/rustlib/${pkgs.stdenv.hostPlatform.rust.rustcTarget}/bin/llvm-cov";
    LLVM_PROFDATA = "${toolchain}/lib/rustlib/${pkgs.stdenv.hostPlatform.rust.rustcTarget}/bin/llvm-profdata";
    TZDIR = "${pkgs.tzdata}/share/zoneinfo";
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS = "-C link-arg=-fuse-ld=mold -C link-arg=-Wl,--compress-debug-sections=zlib";
    shellHook = ''
      # On a host that provides it, each checkout builds in its own target
      # under the shared compiler cache, seeded from a warm target of the same
      # Cargo.lock; an explicitly set target directory is kept.
      if [ -z "''${CARGO_TARGET_DIR:-}" ] && command -v sinnix-cargo-target >/dev/null 2>&1; then
        if cargo_target="$(sinnix-cargo-target)"; then
          export CARGO_TARGET_DIR="$cargo_target"
        fi
      fi
      if [ -d .githooks ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        && ! git config --get core.hooksPath >/dev/null; then
        git config --local core.hooksPath .githooks
      fi
    '';
  };
}
