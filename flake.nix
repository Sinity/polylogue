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
      lib = nixpkgs.lib;
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # CodeQL is unfree in nixpkgs. Keep the exception scoped to the one
        # devshell tool instead of enabling all unfree packages.
        config.allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [ "codeql" ];
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

      # Free-threaded Python 3.14 (PEP 703/779) is the sole interpreter this
      # flake packages (operator decision 2026-07-19, polylogue-xikl: "adopt
      # free-threaded Python across polylogue, fully"). It used to also build
      # a standard (GIL) `polylogue` package + `.#gil` devShell as a parallel
      # deployment target, but that variant had no consumer: sinnix's
      # `polyloguePkg` always pinned `packages.polylogue-freethreaded`
      # (polylogue-dcz5), and nothing else referenced the GIL build outside
      # this flake's own CI checks. Keeping a fully-maintained alternative
      # that ships nowhere is exactly the pattern this project rejects, so
      # the GIL variant is gone and `.#polylogue` -- the attribute CI and
      # sinnix both already resolve -- IS the free-threaded build now, not a
      # separately named alias of it.
      python = pkgs.python314FreeThreading;
      pythonPackages = python.pkgs;

      # Polylogue uses the MCPServer API introduced by MCP 2.x.  nixpkgs'
      # free-threaded package set still supplies MCP 1.x, which imports but
      # has no ``mcp.server.mcpserver`` module and consequently fails only
      # when the stdio server starts.  Keep this small pinned SDK closure in
      # step with the authoritative uv.lock resolution instead of adding an
      # import compatibility layer for two incompatible APIs.
      mkPinnedPythonPackage =
        {
          pname,
          version,
          hash,
          dependencies,
          build-system,
          pythonRelaxDeps ? [ ],
          pythonImportsCheck ? [ ],
        }:
        pythonPackages.buildPythonPackage {
          inherit
            pname
            version
            dependencies
            build-system
            pythonRelaxDeps
            pythonImportsCheck
            ;
          pyproject = true;
          src = pkgs.fetchPypi { inherit pname version hash; };
          env.UV_DYNAMIC_VERSIONING_BYPASS = version;
          doCheck = false;
        };

      uv-dynamic-versioning2 = mkPinnedPythonPackage {
        pname = "uv_dynamic_versioning";
        version = "0.14.1";
        hash = "sha256-hkLbaGzlxQQXA156JXrHO35cOnoywz5FvX42uiLutkg=";
        dependencies = with pythonPackages; [
          dunamai2
          hatchling
          jinja2
          tomlkit
        ];
        build-system = with pythonPackages; [
          hatchling
        ];
      };

      dunamai2 = mkPinnedPythonPackage {
        pname = "dunamai";
        version = "1.26.1";
        hash = "sha256-O0YAe9ZbALSCTq0KGu42X9ItDsK5whlJfU/Uj1KGDIs=";
        dependencies = with pythonPackages; [
          packaging
        ];
        build-system = with pythonPackages; [
          poetry-core
        ];
      };

      httpcore2 = mkPinnedPythonPackage {
        pname = "httpcore2";
        version = "2.12.0";
        hash = "sha256-kpNSK7oKp8TI6ePwQMFldb2IaOFVp3+jDHqQhaXq5kg=";
        dependencies = with pythonPackages; [
          h11
          truststore2
        ];
        build-system = with pythonPackages; [
          hatch-fancy-pypi-readme
          hatchling
          uv-dynamic-versioning2
        ];
      };

      # nixpkgs' truststore derivation currently propagates its documentation
      # and test client stack.  The lockfile's truststore 0.10.4 distribution
      # has no runtime dependencies, so package that authoritative runtime
      # artifact directly for the MCP HTTP client closure.
      truststore2 = mkPinnedPythonPackage {
        pname = "truststore";
        version = "0.10.4";
        hash = "sha256-nZG9Q2RjrV5O5KunZmKN1s1wEM8+JGF1azMDcQ7rwwE=";
        dependencies = [ ];
        build-system = with pythonPackages; [
          flit-core
        ];
      };

      httpx2 = mkPinnedPythonPackage {
        pname = "httpx2";
        version = "2.12.0";
        hash = "sha256-djH+mIeooidfSiVA4FOqZw/MUHQoZKmufGbmCf3PEs8=";
        dependencies = with pythonPackages; [
          anyio
          httpcore2
          idna
          truststore2
        ];
        build-system = with pythonPackages; [
          hatch-fancy-pypi-readme
          hatchling
          uv-dynamic-versioning2
        ];
        pythonRelaxDeps = [ "idna" ];
      };

      mcp-types = mkPinnedPythonPackage {
        pname = "mcp_types";
        version = "2.1.1";
        hash = "sha256-d9y+SPunPMpxpnPyZGpfA3oBe3oKB6yJzsERMCiJDto=";
        dependencies = with pythonPackages; [
          pydantic
          typing-extensions
        ];
        build-system = with pythonPackages; [
          hatchling
          uv-dynamic-versioning2
        ];
      };

      withoutSphinxDocs =
        package:
        package.overridePythonAttrs (old: {
          doCheck = false;
          doInstallCheck = false;
          nativeBuildInputs = builtins.filter (
            input: !(lib.hasInfix "sphinx" (input.pname or (input.name or "")))
          ) (old.nativeBuildInputs or [ ]);
          postInstall = (old.postInstall or "") + ''
            mkdir -p "$doc"
          '';
        });

      # MCP 2.x uses PyJWT's crypto extra.  In the free-threaded package set
      # PyJWT's nixpkgs derivation unconditionally builds Sphinx docs, whose
      # test-only dependency chain is not compatible with Python 3.14t.  The
      # deployed SDK needs the runtime wheel and crypto dependency, not docs.
      pyjwtNoDocs = withoutSphinxDocs pythonPackages.pyjwt;

      # opentelemetry-api reaches the same incompatible Sphinx chain through
      # its runtime dependency `deprecated`.  Rewire that one propagated
      # dependency rather than broadening the MCP SDK closure with test/docs
      # tooling it does not execute.
      wraptNoDocs = withoutSphinxDocs pythonPackages.wrapt;

      # Consumers may follow a different nixpkgs revision. Preserve whichever
      # dependency arguments that revision actually declares, including their
      # Python module metadata, rather than assuming one spelling.
      replacePythonDependency =
        package: dependencyName: replacement:
        package.overridePythonAttrs (
          old:
          lib.genAttrs
            (builtins.filter (field: builtins.hasAttr field old) [
              "dependencies"
              "propagatedBuildInputs"
            ])
            (
              field:
              builtins.map (
                input: if (input.pname or (input.name or "")) == dependencyName then replacement else input
              ) old.${field}
            )
        );

      deprecatedNoDocs =
        replacePythonDependency (withoutSphinxDocs pythonPackages.deprecated) "wrapt"
          wraptNoDocs;

      opentelemetryApiNoDocs =
        replacePythonDependency pythonPackages.opentelemetry-api "deprecated"
          deprecatedNoDocs;

      mcp-sdk = mkPinnedPythonPackage {
        pname = "mcp";
        version = "2.1.1";
        hash = "sha256-ULe6HrvhFwCOp73SiCNAQ+acILQD1oUdGWYebUMade8=";
        dependencies = with pythonPackages; [
          anyio
          cryptography
          httpx2
          jsonschema
          mcp-types
          opentelemetryApiNoDocs
          pydantic
          pyjwtNoDocs
          python-multipart
          sse-starlette
          starlette
          typing-extensions
          typing-inspection
          uvicorn
        ];
        build-system = with pythonPackages; [
          hatchling
          uv-dynamic-versioning2
        ];
        pythonImportsCheck = [
          "mcp.server.mcpserver"
        ];
      };

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

      polylogue = pythonPackages.buildPythonPackage {
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

        # `msgspec` is the sole fast-JSON accelerator: `orjson` ships no
        # cp314t wheel and its build refuses to compile free-threaded, so it
        # can never load on this interpreter (polylogue.core.json's facade
        # falls back to stdlib json when no accelerator is importable; on
        # this build msgspec is always the one it picks).
        dependencies = with pythonPackages; [
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
          mcp-sdk
          pyyaml
          watchfiles
          msgspec
          nh3
        ];

        doCheck = false;
        pythonImportsCheck = [
          "polylogue"
        ];
        dontCheckRuntimeDeps = true;

        # _PYTHON_SYSCONFIGDATA_NAME/_PYTHON_HOST_PLATFORM: a caller devshell
        # for a DIFFERENT interpreter (e.g. a contributor's own system Python)
        # commonly exports these. If either leaks through into this program's
        # env, sysconfig._get_sysconfigdata_name() trusts the inherited value
        # instead of computing its own -- and this build's real module is
        # named with a `t` abiflag segment
        # (`_sysconfigdata_t_linux_x86_64-linux-gnu`) that a non-free-threaded
        # name (`_sysconfigdata__linux_x86_64-linux-gnu`, no `t`) does not
        # match, so any command doing real work (not just --version/--help)
        # fails with `ModuleNotFoundError: No module named
        # '_sysconfigdata__linux_x86_64-linux-gnu'` (polylogue-xikl, reproduced
        # 2026-07-19: PYTHONPATH alone was NOT the trigger -- isolated to this
        # one env var via `env -i` bisection).
        postFixup = ''
          test -f "$out/${python.sitePackages}/polylogue/daemon/static/dist/manifest.json"
          for program in polylogue polylogued polylogue-mcp polylogue-hook; do
            wrapProgram "$out/bin/$program" \
              --unset PYTHONPATH \
              --unset PYTHONHOME \
              --unset PYTHONBREAKPOINT \
              --unset PYTHONUSERBASE \
              --unset VIRTUAL_ENV \
              --unset _PYTHON_SYSCONFIGDATA_NAME \
              --unset _PYTHON_HOST_PLATFORM
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
      # `mkNoCheckOverride` disables `doCheck`/`doInstallCheck` across the
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
      # the standard (GIL) `pkgs.python314Packages` set (nixpkgs' own default),
      # and polylogue's own test suite is unaffected (`doCheck = false` was
      # already set on the `polylogue` derivation itself). Filed as a
      # nixpkgs-upstream gap, not fixed here (out of this lane's packaging
      # scope): see polylogue-xikl for the tracking note.
      mkNoCheckOverride =
        argsOrFn:
        if builtins.isFunction argsOrFn then
          (
            finalAttrs:
            (argsOrFn finalAttrs)
            // {
              doCheck = false;
              doInstallCheck = false;
            }
          )
        else
          (
            argsOrFn
            // {
              doCheck = false;
              doInstallCheck = false;
            }
          );
      # Applied as a top-level `pkgs` overlay (see the `pkgs` binding above) so
      # every package in `python314FreeThreading.pkgs` -- including ones never
      # named here, like `sphinx` or `defusedxml` -- is *constructed* with
      # checks off from the moment nixpkgs' own internal `callPackage`
      # invokes it, keeping sibling dependency resolution self-consistent.
      freeThreadedNoCheckOverlay = _final: prev: {
        # python-discovery 1.4.2 fails two stale assertions under Python
        # 3.14t (test_predicate_with_fallback_specs,
        # test_satisfies_path_not_abs_basename_match). The `.pkgs`
        # replacement below cannot reach it: nixpkgs' own
        # pyproject-version-patch-hook builds its helper env
        # (tomlkit -> poetry-core -> checkInputs virtualenv ->
        # python-discovery) inside the interpreter's INTERNAL package-set
        # fixpoint, which ignores both the replaced `.pkgs` attr and
        # `.override { packageOverrides }` (see the long comment above).
        # `pythonPackagesExtensions` is the one mechanism nixpkgs applies
        # inside every such fixpoint. Scoped to free-threaded interpreters
        # (executable "python3.14t" -- there is no isFreeThreading passthru
        # attr) so the standard 3.12/3.14 sets keep their cache hits.
        # recheck: drop when nixpkgs bumps python-discovery past 1.4.2.
        pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
          (
            _pyFinal: pyPrev:
            prev.lib.optionalAttrs (prev.lib.hasSuffix "t" (pyPrev.python.executable or "")) {
              python-discovery = pyPrev.python-discovery.overrideAttrs (_old: {
                doCheck = false;
                doInstallCheck = false;
              });
              # Next link in the same chain: virtualenv's own test suite
              # fails interpreter discovery under 3.14t once it builds at
              # all (RuntimeError: failed to find interpreter for Builtin
              # discover). Same stale-upstream class, same treatment.
              virtualenv = pyPrev.virtualenv.overrideAttrs (_old: {
                doCheck = false;
                doInstallCheck = false;
              });
              # And its consumer: poetry-core's masonry wheel-tag tests
              # assert ABI tags that do not exist under the free-threaded
              # interpreter (test_tag/test_wheel_c_extension: assert None).
              poetry-core = pyPrev.poetry-core.overrideAttrs (_old: {
                doCheck = false;
                doInstallCheck = false;
              });
            }
          )
        ];
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
              # closure. `dontCheckRuntimeDeps = true` mirrors what
              # `polylogue`'s own derivation already sets for the same class
              # of over-strict wheel-metadata check.
              sqlite-vec = pySuper.sqlite-vec.overrideAttrs (_old: {
                dontCheckRuntimeDeps = true;
              });
              # nixpkgs' own `sse-starlette` package definition (an mcp ->
              # polylogue transitive dependency) lists only `dependencies =
              # [ anyio ]`, but the package's actual wheel metadata declares
              # `starlette` as a required runtime import -- a genuine, narrow
              # nixpkgs packaging gap. Adding it here is the correct fix
              # (starlette is a real runtime need), not merely suppressing
              # the check the way `sqlite-vec` above does.
              sse-starlette = pySuper.sse-starlette.overrideAttrs (old: {
                propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pyFinal.starlette ];
              });
            }
          );
        };
      };

      # Python environment with polylogue pre-installed (for scripting/notebooks).
      polylogueApiPython = python.withPackages (_: [ polylogue ]);

      # Sanitized api-python: the python binary is wrapped with the same env
      # sanitization as the CLI binaries so downstream consumers (sinnix) don't
      # need their own wrapper.
      polylogueApiPythonWrapped =
        pkgs.runCommand "polylogue-api-python-wrapped"
          {
            buildInputs = [ pkgs.makeWrapper ];
          }
          ''
            mkdir -p "$out/bin"
            for f in ${polylogueApiPython}/bin/*; do
              name=$(basename "$f")
              case "$name" in
                python|python3|python3.*)
                  # Same sanitization as the `polylogue` derivation's postFixup
                  # above, including the _PYTHON_SYSCONFIGDATA_NAME/
                  # _PYTHON_HOST_PLATFORM scrub (polylogue-xikl) -- a caller
                  # devshell for a different interpreter can leak these in just
                  # as easily here.
                  makeWrapper "$f" "$out/bin/$name" \
                    --unset PYTHONPATH \
                    --unset PYTHONHOME \
                    --unset PYTHONBREAKPOINT \
                    --unset PYTHONUSERBASE \
                    --unset VIRTUAL_ENV \
                    --unset _PYTHON_SYSCONFIGDATA_NAME \
                    --unset _PYTHON_HOST_PLATFORM
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
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          python
          pkgs.uv
          devtoolsCli
          pkgs.git
          pkgs.ruff
          pkgs.ast-grep
          pkgs.scc
          pkgs.codeql
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          # No PYTHONDONTWRITEBYTECODE here: the cache prefix below already
          # keeps bytecode out of the source tree, and forbidding writes also
          # disables pytest's assertion-rewrite cache, which costs a full
          # re-rewrite of every test module on every run in every xdist worker.
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

          # Install repo git hooks through the shared Git common directory.
          # The helper pins an absolute common-directory path in every
          # worktree's config, so a historical shell hook cannot restore its
          # relative path as the effective route for that worktree.
          git_common_dir=$(git rev-parse --git-common-dir 2>/dev/null || true)
          if [ -n "$git_common_dir" ]; then
            case "$git_common_dir" in
              /*) ;;
              *) git_common_dir="$PWD/$git_common_dir" ;;
            esac
            git_common_dir=$(cd "$git_common_dir" && pwd -P)
            git_common_root=$(cd "$git_common_dir/.." && pwd -P)
            if [ -x "$git_common_root/scripts/configure-git-hooks" ]; then
              "$git_common_root/scripts/configure-git-hooks"
            else
              desired_hooks_path="$git_common_root/.githooks"
              git -C "$git_common_root" config --local extensions.worktreeConfig true
              git -C "$git_common_root" config --local core.hooksPath "$desired_hooks_path"
              while IFS= read -r worktree_path; do
                [ -n "$worktree_path" ] || continue
                [ -d "$worktree_path" ] || continue
                git -C "$worktree_path" config --worktree core.hooksPath "$desired_hooks_path"
              done < <(git -C "$git_common_root" worktree list --porcelain | sed -n 's/^worktree //p')
            fi
          else
            git_common_root="$PWD"
          fi
          # Clean stale __pycache__ dirs under source trees — skip if stamp
          # is fresh (sources haven't changed since last cleanup).
          pyc_stamp=".cache/.last-pyc-cleanup"
          pyc_should_clean=1
          if [ -f "$pyc_stamp" ]; then
            last_clean=$(stat -c %Y "$pyc_stamp" 2>/dev/null || echo 0)
            newest_src=$(find polylogue tests devtools -name '*.py' -printf '%T@\n' 2>/dev/null | sort -rn | sed -n '1p' | cut -d. -f1)
            if [ -n "$newest_src" ] && [ "$last_clean" -ge "$newest_src" ]; then
              pyc_should_clean=0
            fi
          fi
          if [ "$pyc_should_clean" -eq 1 ]; then
            find polylogue tests devtools -type d -name __pycache__ -prune -exec rm -r {} + 2>/dev/null || true
            touch "$pyc_stamp"
          fi

          # The venv must track the devShell interpreter. This previously ran
          # `uv venv` only when .venv was absent, so a venv created against an
          # older interpreter survived a toolchain bump forever and shadowed
          # the nix-provided python on PATH -- recreate whenever the
          # interpreter identity moves (version + free-threaded flag).
          devshell_python="$(command -v python3)"
          devshell_python_id="$("$devshell_python" -c 'import sys; print(sys.version.split()[0], sys._is_gil_enabled())')"
          case "$devshell_python_id" in
            3.14.*\ False) ;;
            *)
              echo "devshell: refusing non-free-threaded CPython 3.14 ($devshell_python_id)" >&2
              return 1
              ;;
          esac
          create_devshell_venv() {
            uv venv --python "$devshell_python"
          }
          venv_python_id=""
          if [ -x .venv/bin/python ]; then
            venv_python_id="$(.venv/bin/python -c 'import sys; print(sys.version.split()[0], getattr(sys, "_is_gil_enabled", lambda: True)())' 2>/dev/null || true)"
          fi
          if [ ! -d .venv ]; then
            echo "devshell: creating virtual environment ($devshell_python_id)" >&2
            create_devshell_venv
          elif [ "$venv_python_id" != "$devshell_python_id" ]; then
            echo "devshell: interpreter changed ($venv_python_id -> $devshell_python_id); recreating .venv" >&2
            rm -rf .venv
            create_devshell_venv
          fi

          # Activate venv
          source .venv/bin/activate
          .venv/bin/python -c 'import sys; assert sys.implementation.name == "cpython" and sys.version_info[:2] >= (3, 14) and not sys._is_gil_enabled(), "Polylogue requires CPython 3.14 free-threading"'

          # Sync dependencies when pyproject.toml, uv.lock, or Python version
          # change. The interpreter component is the devShell's identity
          # captured BEFORE activation -- reading `python --version` here would
          # report the venv that was just activated, which is precisely the
          # value that cannot detect a toolchain bump.
          sync_fingerprint_file=".venv/.uv-sync-fingerprint"
          sync_fingerprint="$(
            cat pyproject.toml uv.lock 2>/dev/null
            printf '%s' "$devshell_python_id"
          )"
          sync_fingerprint="$(printf '%s' "$sync_fingerprint" | sha256sum | cut -d' ' -f1)"
          current_fingerprint=""
          if [ -f "$sync_fingerprint_file" ]; then
            current_fingerprint="$(cat "$sync_fingerprint_file")"
          fi

          if [ "$sync_fingerprint" != "$current_fingerprint" ]; then
            echo "devshell: syncing Python dependencies (fingerprint changed)" >&2
            # Record the fingerprint ONLY after a proven-complete sync: a
            # failed/interrupted sync that still stamped the fingerprint made
            # every later shell (and every dispatched lane) trust a venv with
            # no dev tools — the recurring "missing ruff/pytest in managed
            # env" lane failures.
            if uv sync --extra dev --group audit --frozen --quiet \
               && .venv/bin/python -c "import pytest" 2>/dev/null \
               && [ -x .venv/bin/ruff ]; then
              printf '%s' "$sync_fingerprint" > "$sync_fingerprint_file"
            else
              echo "devshell: dependency sync INCOMPLETE; will retry next shell" >&2
            fi
          fi

          if [[ $- == *i* ]]; then
            devtools status --stderr || true
            export POLYLOGUE_MOTD_RENDERED=1
          fi
        '';
      };

      checks.${system} = {
        default =
          pkgs.runCommand "polylogue-smoke"
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

        # This starts the packaged MCP executable, completes a real stdio
        # initialization request, and then terminates it.  It deliberately
        # uses a fresh archive root and disabled daemon routing: packaging
        # checks must not read or mutate a caller's archive.  In particular,
        # this reaches Polylogue's `from mcp.server.mcpserver import
        # MCPServer` at the production startup seam rather than merely
        # checking that the CLI help text can render.
        mcp-stdio-init =
          pkgs.runCommand "polylogue-mcp-stdio-init"
            {
              nativeBuildInputs = [
                polylogue
              ];
            }
            ''
              export HOME="$TMPDIR/home"
              export XDG_CONFIG_HOME="$TMPDIR/config"
              export XDG_DATA_HOME="$TMPDIR/data"
              export XDG_STATE_HOME="$TMPDIR/state"
              export POLYLOGUE_ARCHIVE_ROOT="$TMPDIR/archive"
              export POLYLOGUE_DAEMON=off
              mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$POLYLOGUE_ARCHIVE_ROOT"

              python - <<'PY'
              import json
              import os
              import select
              import subprocess

              request = {
                  "jsonrpc": "2.0",
                  "id": 1,
                  "method": "initialize",
                  "params": {
                      "protocolVersion": "2025-06-18",
                      "capabilities": {},
                      "clientInfo": {"name": "polylogue-package-check", "version": "1"},
                  },
              }
              process = subprocess.Popen(
                  ["polylogue-mcp"],
                  stdin=subprocess.PIPE,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE,
                  text=True,
                  env=os.environ.copy(),
              )
              try:
                  assert process.stdin is not None
                  assert process.stdout is not None
                  process.stdin.write(json.dumps(request) + "\n")
                  process.stdin.flush()
                  readable, _, _ = select.select([process.stdout], [], [], 60)
                  if not readable:
                      raise RuntimeError("polylogue-mcp did not answer initialize")
                  response = json.loads(process.stdout.readline())
                  if response.get("id") != 1 or "result" not in response:
                      raise RuntimeError(f"unexpected MCP initialize response: {response!r}")
                  if response["result"].get("serverInfo", {}).get("name") != "polylogue":
                      raise RuntimeError(f"unexpected MCP server identity: {response!r}")
              finally:
                  process.terminate()
                  try:
                      process.wait(timeout=10)
                  except subprocess.TimeoutExpired:
                      process.kill()
                      process.wait(timeout=10)
                  if process.returncode not in (0, -15):
                      stderr = process.stderr.read() if process.stderr is not None else ""
                      raise RuntimeError(f"polylogue-mcp exited {process.returncode}: {stderr}")
              PY
              touch $out
            '';

        # Package-level proof (polylogue-6rvt) that the full revision the
        # running artifact reports actually matches this flake's `self`
        # input, not just a plausible-looking string. Fails loudly if a
        # future edit reintroduces truncation (e.g. reverting to
        # `self.shortRev`) or drops the embedded metadata module.
        build-info =
          pkgs.runCommand "polylogue-build-info"
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
                echo "embedded BUILD_DIRTY does not match flake dirty state (${
                  if buildDirty then "True" else "False"
                })" >&2
                cat "$build_info" >&2
                exit 1
              }
              ${
                if buildRevision != "unknown" then
                  ''
                    revision_len=$(printf '%s' "${buildRevision}" | wc -c)
                    [ "$revision_len" -eq 40 ] || {
                      echo "flake revision is not a full 40-character commit hash: ${buildRevision}" >&2
                      exit 1
                    }
                  ''
                else
                  ""
              }
              export HOME=$TMPDIR
              polylogue --version | grep -qF "${builtins.substring 0 8 buildRevision}" || {
                echo "polylogue --version does not surface the short prefix of the embedded revision" >&2
                polylogue --version >&2
                exit 1
              }
              touch $out
            '';

        format =
          pkgs.runCommand "polylogue-format"
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

        lint =
          pkgs.runCommand "polylogue-lint"
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
