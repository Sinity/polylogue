{
  pkgs,
  crane,
  toolchain,
  source,
  revision,
}:
let
  lib = pkgs.lib;
  installGitHooks = ''
    if [ -d .githooks ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
      && ! git config --get core.hooksPath >/dev/null; then
      git config --local core.hooksPath .githooks
    fi
  '';
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
  cleanSource = craneLib.cleanCargoSource source;
  # Build identity belongs to the application, not the dependency cache.
  cargoArtifacts = craneLib.buildDepsOnly (
    common
    // {
      src = cleanSource;
      cargoExtraArgs = "--locked --workspace --all-features";
    }
  );
  # The production graph has no workspace test dependencies or fault
  # features. Resolve exactly the graph the package consumes, so changing
  # source does not recompile dependencies under another feature set.
  packageArgs = "--locked -p polylogue --features polylogue/sinex";
  packageArtifacts = craneLib.buildDepsOnly (
    common
    // {
      src = cleanSource;
      cargoExtraArgs = packageArgs;
      # The package consumes codegen artifacts, not cargo-check metadata.
      buildPhaseCargoCommand = "cargoWithProfile build ${packageArgs}";
      doCheck = false;
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
      cargoArtifacts = packageArtifacts;
      cargoExtraArgs = packageArgs;
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
        node ${../extension}/scripts/validate-manifest.mjs
        node ${../extension}/scripts/build.mjs --out "$out"
        mkdir -p "$out/share/polylogue/extension"
        chrome=$(jq -r '.artifacts.chrome' "$out/build-manifest.json")
        unzip -q "$out/$chrome" -d "$out/share/polylogue/extension"
      '';
  checks = {
    settings = import ./checks/settings.nix { inherit pkgs; };
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
        cargoExtraArgs = "--locked --workspace --exclude polylogue-sinex --features polylogue-model/fixture,polylogue-query/fixture,polylogue-store/fixture,polylogue-engine/faults,polylogue-store/faults";
        # Shell completion stays offline; the remaining real-process tests need
        # an OS identity. Checks outside Nix keep every e2e; the VM covers transports.
        cargoNextestExtraArgs = "-E 'not (package(polylogue) and binary(e2e) and not test(/^completion::/))'";
        nativeBuildInputs = common.nativeBuildInputs ++ [
          pkgs.git
          pkgs.python3
          pkgs.dbus
          pkgs.nodejs_22
          pkgs.sqlite
          pkgs.zsh
          pkgs.fish
          (lib.getBin pkgs.util-linux)
        ];
        SHELL = pkgs.runtimeShell;
        TZDIR = "${pkgs.tzdata}/share/zoneinfo";
        preCheck = ''
          export XDG_STATE_HOME="$TMPDIR/state"
        '';
      }
    );

    inherit extension;
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
    polylogue-dependencies = packageArtifacts;
    polylogue-extension = extension;
    default = polylogue;
  };
  inherit checks;
  formatShell = pkgs.mkShellNoCC {
    packages = [
      toolchain
      pkgs.git
      pkgs.diffutils
    ];
    shellHook = installGitHooks;
  };
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
      pkgs.zip
      pkgs.gcc
      pkgs.pkg-config
      pkgs.zstd
      pkgs.dbus
      pkgs.sqlite
      pkgs.zsh
      pkgs.fish
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
    ''
    + installGitHooks;
  };
}
