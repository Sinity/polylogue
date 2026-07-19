{
  description = "Polylogue - local evidence system for AI work";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # See the long comment above `mkNoCheckOverride`/`freeThreadedNoCheckOverlay`
        # below (polylogue-xikl): this MUST be a top-level `pkgs` overlay, not a
        # locally-scoped `.pkgs.overrideScope` on a `let`-bound variable --
        # transitive dependency resolution for packages pulled in via ANOTHER
        # python314FreeThreading package's own `nativeBuildInputs` (e.g. `sphinx`
        # via `pyjwt`) goes through nixpkgs' own internal fixed point, which a
        # locally-scoped override never reaches (proven by a failed build:
        # local-only scoping left `sphinx`/`defusedxml` resolving to the
        # original, still-broken derivations).
        overlays = [ freeThreadedNoCheckOverlay ];
      };
      python = pkgs.python314;

      # Script body lives in nix/devtools-wrapper.sh so it can be unit-tested
      # directly (see tests/unit/devtools/test_cli_wrapper.py).
      devtoolsCli = pkgs.writeShellScriptBin "devtools" (builtins.readFile ./nix/devtools-wrapper.sh);

      # Full immutable git revision embedded at build time (polylogue-6rvt).
      #
      # `self.rev`/`self.dirtyRev` (not `self.shortRev`/`self.dirtyShortRev`)
      # so the packaged runtime can independently attest its exact source
      # identity against a consuming flake's `flake.lock` `rev`/`narHash`
      # entry for this input, not just a truncated, collision-prone prefix.
      # `self.dirtyRev` carries a literal "-dirty" suffix baked into the
      # string; strip it so `buildRevision` is always either a clean
      # 40-character hex commit or the explicit sentinel "unknown", with
      # dirtiness tracked separately in `buildDirty`.
      buildDirty = self ? dirtyRev;
      buildRevision =
        if self ? rev then
          self.rev
        else if self ? dirtyRev then
          pkgs.lib.removeSuffix "-dirty" self.dirtyRev
        else
          "unknown";

      # Shared package builder for both the standard (GIL) interpreter and the
      # free-threaded (3.14t) variant (polylogue-xikl phase-0/deployment-edge:
      # CLI-first rebuild lane on 3.14t while the daemon stays standard). The
      # two variants differ ONLY in which interpreter/package set they build
      # against and which fast-JSON accelerator they carry -- orjson has no
      # cp314t wheels and its build refuses to compile free-threaded, so the
      # free-threaded variant carries `msgspec` instead (polylogue.core.json's
      # facade picks whichever backend is importable, no code change needed).
      mkPolylogue =
        { python', pythonPackages, jsonAccelerator }:
        pythonPackages.buildPythonPackage {
        pname = "polylogue";
        # Single authoritative version: pyproject.toml (release-please owns bumps).
        version = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
        pyproject = true;
        src = ./.;

        postPatch = ''
          cat > polylogue/_build_info.py << BUILDEOF
          BUILD_COMMIT = "${buildRevision}"
          BUILD_DIRTY = ${if buildDirty then "True" else "False"}
          BUILDEOF
        '';

        build-system = with pythonPackages; [
          hatchling
        ];

        nativeBuildInputs = [
          pkgs.makeWrapper
        ];

        dependencies = (with pythonPackages; [
          google-auth-oauthlib
          google-api-python-client
          google-auth-httplib2
          httpx
          h2
          rich
          textual
          jinja2
          markdown-it-py
          pygments
          ijson
          lark
          sqlite-vec
          questionary
          click
          tenacity
          dateparser
          structlog
          pydantic
          aiosqlite
          mcp
          pyyaml
          watchfiles
        ]) ++ [ jsonAccelerator ];

        doCheck = false;
        pythonImportsCheck = [
          "polylogue"
        ];
        dontCheckRuntimeDeps = true;

        postFixup = ''
          test -f "$out/${python'.sitePackages}/polylogue/daemon/static/dist/manifest.json"
          for program in polylogue polylogued polylogue-mcp; do
            wrapProgram "$out/bin/$program" \
              --unset PYTHONPATH \
              --unset PYTHONHOME \
              --unset PYTHONBREAKPOINT \
              --unset PYTHONUSERBASE \
              --unset VIRTUAL_ENV
          done
        '';

        meta = {
          description = "Polylogue evidence archive Python package and CLI";
          mainProgram = "polylogue";
          license = pkgs.lib.licenses.mit;
          homepage = "https://github.com/Sinity/polylogue";
          platforms = pkgs.lib.platforms.linux;
        };
      };

      polylogue = mkPolylogue {
        python' = python;
        pythonPackages = pkgs.python314Packages;
        jsonAccelerator = pkgs.python314Packages.orjson;
      };

      # Free-threaded (3.14t, PEP 779) build variant (polylogue-xikl /
      # polylogue-7mtf): same package, built against
      # `python314FreeThreading.pkgs` so the CLI can run with the GIL
      # disabled. This is the offline/bulk-rebuild deployment edge -- the
      # daemon stays on the standard build until the free-threading adoption
      # gate + thread-safety audit (phase 1/3 of polylogue-xikl) land.
      #
      # `python314FreeThreading` is uncached at this nixpkgs pin (2026-07-19):
      # nothing in its `.pkgs` set has a prebuilt binary on cache.nixos.org
      # yet, so *every* Python package in the closure builds from source with
      # its upstream `doCheck`/`doInstallCheck` default (true), pulling each
      # package's own test-only `checkInputs` transitively. In practice that
      # fans out to ~500 unrelated packages (django, matplotlib, scipy,
      # mercurial, sphinx, ...) and hits real, unrelated packaging bugs in
      # that long tail that have nothing to do with polylogue itself: e.g.
      # nixpkgs' `boost` built with Python bindings against the free-threaded
      # ABI fails outright (`wrap_python.hpp: pyconfig.h: No such file or
      # directory` -- boost 1.89.0's python integration doesn't know the
      # cp314t layout yet), and `defusedxml`'s own installCheckPhase fails
      # against Python 3.14's stdlib (`gzip.GzipFile.__del__` AttributeError +
      # a DeprecationWarning -> RuntimeWarning category change) -- both
      # cascade to every package whose checkInputs pull them in transitively
      # (matplotlib/pytest-mpl/sphinx-pytest et al., pulled in only to run
      # OTHER packages' test suites, not by anything polylogue needs at
      # runtime).
      #
      # `pythonFreeThreadedPkgs` disables `doCheck`/`doInstallCheck` across the
      # whole package set by wrapping `buildPythonPackage`/
      # `buildPythonApplication` themselves via `overrideScope`, rather than
      # patching already-built package derivations after the fact. Two other
      # approaches were tried and empirically rejected first (verified via
      # `nix derivation show`/`.drvPath` comparisons before running the full
      # build, since each is a ~15-30 minute source build to disprove):
      #   1. `python314FreeThreading.override { packageOverrides = ...; }` on
      #      the interpreter itself silently no-ops here -- `.drvPath` was
      #      byte-identical before/after, so nothing was applied at all.
      #   2. `pkgs.overrideScope (final: prev: mapAttrs (_: drv: drv.
      #      overridePythonAttrs (_: {doCheck=false;...})) prev)` DOES change
      #      the *targeted* package's own hash/env, but `overridePythonAttrs`
      #      patches an already-constructed derivation whose OWN
      #      `dependencies`/`nativeBuildInputs` list is a fixed reference to
      #      sibling packages as originally composed -- so a downstream
      #      package like `sphinx` still built against the *original* (still
      #      failing) `defusedxml`, even though `sphinx`'s own doCheck flag
      #      flipped. The cascade only breaks when the failing leaf's
      #      `nativeCheckInputs` are dropped as part of *constructing*
      #      `sphinx` with `doCheck=false` from the start, not retrofitted.
      # Wrapping the shared `buildPythonPackage`/`buildPythonApplication`
      # builder functions instead means every package -- including ones this
      # override never names, like `sphinx` or `defusedxml` -- is constructed
      # with checks off from the moment `callPackage` invokes it, so sibling
      # dependency resolution stays self-consistent. Confirmed via `nix
      # derivation show`: `sphinx`'s own `nativeCheckInputs` becomes `[]` and
      # its hash changes accordingly. The wrapper handles both
      # `buildPythonPackage` calling conventions (a plain attrset, and the
      # newer `finalAttrs: {...}` self-referencing function -- naively doing
      # `args // {...}` on the latter throws "expected a set but found a
      # function", hit and fixed while developing this).
      #
      # Packaging-only: upstream test suites for these packages still run on
      # the standard (GIL) build via `pkgs.python314Packages`, and polylogue's
      # own test suite is unaffected (`doCheck = false` was already set on the
      # `polylogue` derivation itself, standard build included). Filed as a
      # nixpkgs-upstream gap, not fixed here (out of this lane's packaging
      # scope): see polylogue-xikl for the tracking note.
      mkNoCheckOverride =
        argsOrFn:
        if builtins.isFunction argsOrFn then
          (finalAttrs: (argsOrFn finalAttrs) // { doCheck = false; doInstallCheck = false; })
        else
          (argsOrFn // { doCheck = false; doInstallCheck = false; });
      # Applied as a top-level `pkgs` overlay (see the `pkgs` binding above) so
      # every package in `python314FreeThreading.pkgs` -- including ones never
      # named here, like `sphinx` or `defusedxml` -- is *constructed* with
      # checks off from the moment nixpkgs' own internal `callPackage`
      # invokes it, keeping sibling dependency resolution self-consistent.
      freeThreadedNoCheckOverlay = _final: prev: {
        python314FreeThreading = prev.python314FreeThreading // {
          pkgs = prev.python314FreeThreading.pkgs.overrideScope (
            pyFinal: pySuper: {
              buildPythonPackage = argsOrFn: pySuper.buildPythonPackage (mkNoCheckOverride argsOrFn);
              buildPythonApplication = argsOrFn: pySuper.buildPythonApplication (mkNoCheckOverride argsOrFn);
              # `pyjwt` (an mcp -> polylogue transitive dependency) unconditionally
              # builds Sphinx-based docs (`nativeBuildInputs = [ sphinxHook
              # sphinx-rtd-theme ... ]`, `outputs = [ "out" "doc" ]`) -- this is
              # NOT gated by doCheck at all, so the builder-wrap above doesn't
              # touch it, and it's the ONLY path from polylogue into the
              # sphinx -> defusedxml chain (defusedxml's own installCheckPhase
              # fails against Python 3.14's stdlib, see the long comment
              # above). Strip the sphinx build tools from nativeBuildInputs;
              # the CLI never needs pyjwt's docs. Two follow-on breakages hit
              # and fixed while developing this: (1) forcing `outputs` down to
              # `[ "out" ]` broke `pythonOutputDistPhase`'s always-present
              # `dist` output wiring ("mv: cannot move 'dist' to '': Device or
              # resource busy") -- so `outputs` is left untouched, still
              # `[ "out" "doc" ]`; (2) but removing sphinxHook means nothing
              # ever creates the `$doc` output path at all, so nix then fails
              # with "failed to produce output path for output 'doc'" --
              # fixed by explicitly `mkdir -p $doc` in `postInstall` (an
              # empty doc output is harmless; nothing consumes it).
              pyjwt = pySuper.pyjwt.overrideAttrs (old: {
                doCheck = false;
                doInstallCheck = false;
                nativeBuildInputs = builtins.filter (
                  # `hasInfix`, not `hasPrefix`: the setup-hook derivation's
                  # own name is "python3.14-sphinx-hook" (prefixed by the
                  # STANDARD interpreter version, not "sphinx"), so a prefix
                  # check misses it while still catching "sphinx-rtd-theme"
                  # -- caught empirically when the first attempt (hasPrefix)
                  # left sphinx-hook in place and the build still cascaded.
                  i: !(prev.lib.hasInfix "sphinx" (i.pname or (i.name or "")))
                ) old.nativeBuildInputs;
                postInstall = (old.postInstall or "") + ''
                  mkdir -p "$doc"
                '';
              });
              # `sqlite-vec`'s wheel metadata lists an optional `numpy` extra
              # (a convenience helper for feeding numpy arrays into
              # `serialize_float32`; polylogue never uses it -- the package is
              # otherwise a thin ctypes wrapper loading a self-contained
              # SQLite `.so` extension, ABI-independent of the Python build).
              # nixpkgs' `pythonRuntimeDepsCheckHook` fails the build because
              # this free-threaded package set has no `numpy` in this
              # closure; the standard (GIL) build silently passes only
              # because numpy happens to already be resolvable there for
              # unrelated reasons. `dontCheckRuntimeDeps = true` mirrors what
              # `polylogue`'s own derivation already sets for the same class
              # of over-strict wheel-metadata check.
              sqlite-vec = pySuper.sqlite-vec.overrideAttrs (_old: {
                dontCheckRuntimeDeps = true;
              });
              # nixpkgs' own `sse-starlette` package definition (an mcp ->
              # polylogue transitive dependency) lists only `dependencies =
              # [ anyio ]`, but the package's actual wheel metadata declares
              # `starlette` as a required runtime import -- a genuine, narrow
              # nixpkgs packaging gap (present for the standard build too;
              # just never triggers the failure there because `mcp`'s other
              # direct dependencies already happen to pull `starlette` into
              # that build's closure). Adding it here is the correct fix
              # (starlette is a real runtime need), not merely suppressing
              # the check the way `sqlite-vec` above does.
              sse-starlette = pySuper.sse-starlette.overrideAttrs (old: {
                propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pyFinal.starlette ];
              });
            }
          );
        };
      };
      pythonFreeThreaded = pkgs.python314FreeThreading;
      polylogueFreeThreaded = mkPolylogue {
        python' = pythonFreeThreaded;
        pythonPackages = pythonFreeThreaded.pkgs;
        jsonAccelerator = pythonFreeThreaded.pkgs.msgspec;
      };

      # Python environment with polylogue pre-installed (for scripting/notebooks).
      polylogueApiPython = python.withPackages (_: [ polylogue ]);

      # Sanitized api-python: the python binary is wrapped with the same env
      # sanitization as the CLI binaries so downstream consumers (sinnix) don't
      # need their own wrapper.
      polylogueApiPythonWrapped = pkgs.runCommand "polylogue-api-python-wrapped"
        {
          buildInputs = [ pkgs.makeWrapper ];
        }
        ''
          mkdir -p "$out/bin"
          for f in ${polylogueApiPython}/bin/*; do
            name=$(basename "$f")
            case "$name" in
              python|python3|python3.*)
                makeWrapper "$f" "$out/bin/$name" \
                  --unset PYTHONPATH \
                  --unset PYTHONHOME \
                  --unset PYTHONBREAKPOINT \
                  --unset PYTHONUSERBASE \
                  --unset VIRTUAL_ENV
                ;;
              *)
                ln -s "$f" "$out/bin/$name"
                ;;
            esac
          done
          for d in lib include share; do
            if [ -d "${polylogueApiPython}/$d" ]; then
              ln -s "${polylogueApiPython}/$d" "$out/$d"
            fi
          done
        '';
    in
    {
      packages.${system} = {
        inherit polylogue;
        default = polylogue;
        api-python = polylogueApiPythonWrapped;
        api-python-raw = polylogueApiPython;
        polylogue-freethreaded = polylogueFreeThreaded;
      };

      devShells.${system} = {
      default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python
          uv
          devtoolsCli
          git
          ruff
          py-spy
          python.pkgs.pyinstrument
          bat
          glow
          openssl
          vhs
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          export HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.cache/hypothesis"
          export PYTHONDONTWRITEBYTECODE=1
          export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
          export POLYLOGUE_REPO_ROOT="$PWD"
          mkdir -p .cache .local "$PYTHONPYCACHEPREFIX"

          if [ -L result ]; then
            rm result
          fi

          # One-time cleanup of legacy repo-root cache dirs (migrated under .cache/).
          legacy_stamp=".cache/.legacy-caches-migrated"
          if [ ! -f "$legacy_stamp" ]; then
            for legacy_cache_root in __pycache__ .pytest_cache .hypothesis .mypy_cache .ruff_cache .benchmarks; do
              if [ -e "$legacy_cache_root" ]; then
                rm -rf "$legacy_cache_root"
              fi
            done
            touch "$legacy_stamp"
          fi

          # Install repo git hooks — skip if already set correctly.
          # Prefer the Beads composite hook path when present; those hooks
          # chain the ordinary Polylogue .githooks checks and the Beads hooks.
          desired_hooks_path=".githooks"
          if [ -d .beads-hooks ]; then
            desired_hooks_path=".beads-hooks"
          fi
          current_hooks_path=$(git config --local core.hooksPath 2>/dev/null || true)
          if [ "$current_hooks_path" != "$desired_hooks_path" ]; then
            git config --local core.hooksPath "$desired_hooks_path" 2>/dev/null || true
          fi

          # Clean stale __pycache__ dirs under source trees — skip if stamp
          # is fresh (sources haven't changed since last cleanup).
          pyc_stamp=".cache/.last-pyc-cleanup"
          pyc_should_clean=1
          if [ -f "$pyc_stamp" ]; then
            last_clean=$(stat -c %Y "$pyc_stamp" 2>/dev/null || echo 0)
            newest_src=$(find polylogue tests devtools -name '*.py' -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)
            if [ -n "$newest_src" ] && [ "$last_clean" -ge "$newest_src" ]; then
              pyc_should_clean=0
            fi
          fi
          if [ "$pyc_should_clean" -eq 1 ]; then
            find polylogue tests devtools -type d -name __pycache__ -prune -exec rm -r {} + 2>/dev/null || true
            touch "$pyc_stamp"
          fi

          # Create venv if it doesn't exist
          if [ ! -d .venv ]; then
            echo "devshell: creating virtual environment" >&2
            uv venv
          fi

          # Activate venv
          source .venv/bin/activate

          # Sync dependencies when pyproject.toml, uv.lock, or Python version change.
          sync_fingerprint_file=".venv/.uv-sync-fingerprint"
          sync_fingerprint="$(
            cat pyproject.toml uv.lock 2>/dev/null
            python --version 2>&1
          )"
          sync_fingerprint="$(printf '%s' "$sync_fingerprint" | sha256sum | cut -d' ' -f1)"
          current_fingerprint=""
          if [ -f "$sync_fingerprint_file" ]; then
            current_fingerprint="$(cat "$sync_fingerprint_file")"
          fi

          if [ "$sync_fingerprint" != "$current_fingerprint" ]; then
            echo "devshell: syncing Python dependencies (fingerprint changed)" >&2
            uv sync --extra dev --frozen --quiet
            printf '%s' "$sync_fingerprint" > "$sync_fingerprint_file"
          fi

          if [[ $- == *i* ]]; then
            devtools status --stderr || true
            export POLYLOGUE_MOTD_RENDERED=1
          fi
        '';
      };

      # Free-threaded (3.14t) devShell variant (polylogue-xikl / polylogue-7mtf
      # experiment gate). Deliberately lean: it provides the free-threaded
      # interpreter for interactive smoke/benchmark work, but it does NOT
      # auto-sync `--extra dev` into a venv the way the standard shell does,
      # because pyproject.toml's `dev` extra pulls in `orjson` unconditionally
      # -- orjson has no cp314t wheels and its build refuses to compile
      # free-threaded, so a frozen sync targeting this interpreter would fail
      # outright. Running the unit suite here today means a manual,
      # non-frozen `uv pip install` of the test tooling (pytest/hypothesis/
      # etc.) plus `msgspec` in place of `orjson`; a proper `dev-freethreaded`
      # pyproject extra + uv.lock entry is tracked as follow-up, not done
      # here to keep this lane scoped to packaging.
      freethreaded = pkgs.mkShell {
        buildInputs = with pkgs; [
          pythonFreeThreaded
          uv
          devtoolsCli
          git
          ruff
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          export PYTHONDONTWRITEBYTECODE=1
          export POLYLOGUE_REPO_ROOT="$PWD"
          echo "devshell(freethreaded): $(python3 -c 'import sys; print(sys.version, "GIL enabled:", sys._is_gil_enabled())')" >&2
          echo "devshell(freethreaded): venv not auto-synced -- see flake.nix comment above this shell for the manual setup" >&2
        '';
      };
      };

      checks.${system} = {
        default = pkgs.runCommand "polylogue-smoke"
          {
            nativeBuildInputs = [
              polylogue
            ];
          }
          ''
            export HOME=$TMPDIR
            polylogue --help >/dev/null
            polylogued --help >/dev/null
            polylogue-mcp --help >/dev/null
            touch $out
          '';

        # Package-level proof (polylogue-6rvt) that the full revision the
        # running artifact reports actually matches this flake's `self`
        # input, not just a plausible-looking string. Fails loudly if a
        # future edit reintroduces truncation (e.g. reverting to
        # `self.shortRev`) or drops the embedded metadata module.
        build-info = pkgs.runCommand "polylogue-build-info"
          {
            nativeBuildInputs = [
              polylogue
            ];
          }
          ''
            build_info="${polylogue}/${python.sitePackages}/polylogue/_build_info.py"
            echo "checking $build_info" >&2
            grep -qxF 'BUILD_COMMIT = "${buildRevision}"' "$build_info" || {
              echo "embedded BUILD_COMMIT does not match flake self.rev/self.dirtyRev (${buildRevision})" >&2
              cat "$build_info" >&2
              exit 1
            }
            grep -qxF 'BUILD_DIRTY = ${if buildDirty then "True" else "False"}' "$build_info" || {
              echo "embedded BUILD_DIRTY does not match flake dirty state (${if buildDirty then "True" else "False"})" >&2
              cat "$build_info" >&2
              exit 1
            }
            ${if buildRevision != "unknown" then ''
              revision_len=$(printf '%s' "${buildRevision}" | wc -c)
              [ "$revision_len" -eq 40 ] || {
                echo "flake revision is not a full 40-character commit hash: ${buildRevision}" >&2
                exit 1
              }
            '' else ""}
            export HOME=$TMPDIR
            polylogue --version | grep -qF "${builtins.substring 0 8 buildRevision}" || {
              echo "polylogue --version does not surface the short prefix of the embedded revision" >&2
              polylogue --version >&2
              exit 1
            }
            touch $out
          '';

        format = pkgs.runCommand "polylogue-format"
          {
            nativeBuildInputs = [
              pkgs.ruff
            ];
          }
          ''
            export RUFF_CACHE_DIR=$TMPDIR/.ruff-cache
            cd ${self}
            ruff format --check polylogue/ tests/ devtools/
            touch $out
          '';

        lint = pkgs.runCommand "polylogue-lint"
          {
            nativeBuildInputs = [
              pkgs.ruff
            ];
          }
          ''
            export RUFF_CACHE_DIR=$TMPDIR/.ruff-cache
            cd ${self}
            ruff check polylogue/ tests/ devtools/
            touch $out
          '';
      };

      formatter.${system} = pkgs.nixfmt;

      apps.${system} = {
        polylogue = {
          type = "app";
          program = "${polylogue}/bin/polylogue";
        };
        polylogued = {
          type = "app";
          program = "${polylogue}/bin/polylogued";
        };
        polylogue-mcp = {
          type = "app";
          program = "${polylogue}/bin/polylogue-mcp";
        };
        default = {
          type = "app";
          program = "${polylogue}/bin/polylogue";
        };
      };

      nixosModules.default = import ./nix/module.nix;
      nixosModules.polylogue = import ./nix/module.nix;
      homeManagerModules.default = import ./nix/hm-module.nix;
      homeManagerModules.polylogue = import ./nix/hm-module.nix;
      homeManagerModules.agentIntegration = import ./nix/agent-integration-module.nix;
    };
}
