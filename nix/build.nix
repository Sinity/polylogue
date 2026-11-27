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
  # sccache hashes every CARGO_* variable, so a target directory chosen per
  # checkout through CARGO_TARGET_DIR would make every checkout miss the
  # others' entries; the compiler never reads it. A server's start creates a
  # socket in TMPDIR, which fails past the socket path limit, so a long TMPDIR
  # is reached through a short link to it under the runtime directory.
  sccacheRustc = pkgs.writeShellScript "sccache-rustc" ''
    unset CARGO_TARGET_DIR
    if [ "''${#TMPDIR}" -gt 60 ] && [ -n "''${XDG_RUNTIME_DIR:-}" ]; then
      short="$XDG_RUNTIME_DIR/sccache-tmp-$(printf %s "$TMPDIR" | ${pkgs.coreutils}/bin/sha256sum | ${pkgs.coreutils}/bin/cut -c1-16)"
      [ -L "$short" ] || ${pkgs.coreutils}/bin/ln -sfn "$TMPDIR" "$short" 2>/dev/null || true
      [ ! -L "$short" ] || export TMPDIR="$short"
    fi
    exec ${pkgs.sccache}/bin/sccache "$@"
  '';

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
  # The dependency build compiles exactly the graph the package consumes
  # (its package, features and profile, no test targets), so a change to
  # Polylogue's own source recompiles only workspace crates.
  packageArgs = "--locked -p polylogue --features polylogue/sinex";
  packageArtifacts = craneLib.buildDepsOnly (
    common
    // {
      src = craneLib.cleanCargoSource source;
      cargoExtraArgs = packageArgs;
      # The package consumes codegen artifacts, not cargo-check metadata.
      buildPhaseCargoCommand = "cargoWithProfile build ${packageArgs}";
      doCheck = false;
    }
  );
  # The binary runs the git it finds on PATH; the deployment modules add a
  # fallback after the user's PATH (lib/settings.nix).
  polylogue = craneLib.buildPackage (
    common
    // {
      # Build identity belongs to the application, not the dependency cache.
      POLYLOGUE_REVISION = revision;
      cargoArtifacts = packageArtifacts;
      cargoExtraArgs = packageArgs;
      doCheck = false;
      doInstallCargoArtifacts = false;
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
        node ${../extension}/scripts/build.mjs --revision ${lib.escapeShellArg revision} --out "$out"
        mkdir -p "$out/share/polylogue/extension"
        chrome=$(jq -r '.artifacts.chrome' "$out/build-manifest.json")
        unzip -q "$out/$chrome" -d "$out/share/polylogue/extension"
        jq -er '.chrome_extension_id' "$out/build-manifest.json" > "$out/share/polylogue/extension-id"
      '';
  checks = {
    settings = import ./checks/settings.nix { inherit pkgs; };
    inherit polylogue;
    fmt = craneLib.cargoFmt {
      inherit (common) src pname version;
      cargoExtraArgs = "--all";
    };
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
      pkgs.cargo-hakari
      pkgs.sccache
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
          # One compiler cache for every checkout's target, beside them.
          : "''${SCCACHE_DIR:=$(dirname "$(dirname "$cargo_target")")/sccache}"
          export SCCACHE_DIR
        fi
      fi
      # Where the host names a compiler cache, compilations go through
      # sccache, with one server per checkout on a socket under the runtime
      # directory. A job lasts until its last process exits, so a server
      # started by a compilation stops 10 s after its last one; xtask starts
      # and stops its own around a run. A server that is gone costs the next
      # compilation a cache lookup, not a failure.
      if [ -n "''${SCCACHE_DIR:-}" ] && [ -z "''${RUSTC_WRAPPER:-}" ] && [ -n "''${XDG_RUNTIME_DIR:-}" ]; then
        export RUSTC_WRAPPER=${sccacheRustc}
        export SCCACHE_IGNORE_SERVER_IO_ERROR=1
        export SCCACHE_IDLE_TIMEOUT=10
        export SCCACHE_CACHE_SIZE="''${SCCACHE_CACHE_SIZE:-4G}"
        SCCACHE_SERVER_UDS="$XDG_RUNTIME_DIR/sccache-$(printf %s "$PWD" | sha256sum | cut -c1-16).sock"
        export SCCACHE_SERVER_UDS
      fi
    ''
    + installGitHooks;
  };
}
