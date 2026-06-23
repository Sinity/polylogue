# Shared option tree + TOML-rendering for polylogue's NixOS and HM modules.
#
# Both ``nix/module.nix`` (system unit) and ``nix/hm-module.nix`` (user
# unit) consume this so the polylogue.toml schema stays in lockstep
# between deployment modes. Downstream modules (e.g. sinnix) can also
# import the helpers if they want to render a config file outside the
# bundled modules — though prefer importing the bundled modules and
# extending via standard NixOS option merging.
{ lib, pkgs }:
let
  inherit (lib) mkOption types;

  # Canonical source-root paths for known providers. Used by the
  # ``discoverSources`` convenience option so downstreams don't have
  # to memorize agent storage layouts.
  knownProviderRoots = {
    claude = "$HOME/.claude/projects";
    claude-code = "$HOME/.claude/projects";
    codex = "$HOME/.codex/sessions";
    gemini = "$HOME/.gemini/tmp";
    antigravity = "$HOME/.gemini/antigravity";
    hermes = "$HOME/.hermes/sessions";
  };

  providerNames = lib.attrNames knownProviderRoots;

  settingsOptions = {
    archive = {
      root = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Archive root directory.";
      };
    };

    daemon = {
      host = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Legacy daemon listen host alias (polylogue default: 127.0.0.1).";
      };
      port = mkOption {
        type = types.nullOr types.port;
        default = null;
        description = "Legacy daemon HTTP API port alias (polylogue default: 8766).";
      };
      watch = mkOption {
        type = types.nullOr (types.listOf types.str);
        default = null;
        description = ''
          Additional source roots for live ingestion. Rendered as
          ``[sources] roots`` in polylogue.toml and merged with any
          paths produced by ``discoverSources``.
        '';
      };
      debounce-s = mkOption {
        type = types.nullOr types.number;
        default = null;
        description = "Live watcher debounce in seconds (polylogue default: 2.0).";
      };
    };

    daemon-api = {
      host = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "API listen host.";
      };
      port = mkOption {
        type = types.nullOr types.port;
        default = null;
        description = "API listen port (polylogue default: 8766).";
      };
      auth-token = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = ''
          API Bearer auth token. Prefer sourcing this from a secret manager
          into ``POLYLOGUE_API_AUTH_TOKEN``; setting it here renders it into
          the generated TOML.
        '';
      };
    };

    browser-capture = {
      host = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Browser-capture receiver host (polylogue default: 127.0.0.1).";
      };
      port = mkOption {
        type = types.nullOr types.port;
        default = null;
        description = "Browser-capture receiver port (polylogue default: 8765).";
      };
      allowed-origins = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Comma-separated allowed CORS origins (default: chrome-extension://*).";
      };
      allow-remote = mkOption {
        type = types.nullOr types.bool;
        default = null;
        description = "Explicitly allow non-loopback browser-capture/API binding.";
      };
      auth-token = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = ''
          Browser-capture Bearer auth token. Required when remote binding or
          non-extension web origins are configured. Prefer an environment
          secret over rendering this into TOML.
        '';
      };
      spool-path = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Browser-capture spool path.";
      };
    };

    embedding = {
      enabled = mkOption {
        type = types.nullOr types.bool;
        default = null;
        description = "Enable post-ingest embedding generation.";
      };
      model = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Voyage AI model name.";
      };
      dimension = mkOption {
        type = types.nullOr types.ints.unsigned;
        default = null;
        description = "Embedding dimension (default: 1024).";
      };
      max-cost-usd = mkOption {
        type = types.nullOr types.number;
        default = null;
        description = "Maximum USD cost for embeddings (0 = unlimited).";
      };
      voyage-api-key = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = ''
          Voyage API key. Prefer ``VOYAGE_API_KEY`` from a secret manager;
          setting this here renders it into the generated TOML.
        '';
      };
    };

    observability = {
      enabled = mkOption {
        type = types.nullOr types.bool;
        default = null;
        description = "Enable OTLP/observability HTTP ingestion routes.";
      };
      otlp-max-body-bytes = mkOption {
        type = types.nullOr types.ints.unsigned;
        default = null;
        description = "Maximum accepted OTLP request body size.";
      };
    };

    logging = {
      level = mkOption {
        type = types.nullOr (types.enum [ "DEBUG" "INFO" "WARNING" "ERROR" ]);
        default = null;
        description = "Log level.";
      };
      force-plain = mkOption {
        type = types.nullOr types.bool;
        default = null;
        description = "Force plain (non-rich) output.";
      };
    };

    ui = {
      theme = mkOption {
        type = types.nullOr (types.enum [ "dark" "light" "auto" ]);
        default = null;
        description = "Semantic theme mode.";
      };
      slow-query-notice-seconds = mkOption {
        type = types.nullOr types.number;
        default = null;
        description = "Threshold for slow-query user notices.";
      };
    };

    schema = {
      validation = mkOption {
        type = types.nullOr (types.enum [ "off" "advisory" "strict" ]);
        default = null;
        description = "Schema validation mode.";
      };
    };

    notifications = {
      backend = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Notification backend: log/stdout/webhook/apprise/email.";
      };
      webhook-url = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Notification webhook URL.";
      };
      webhook-secret = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Notification webhook secret; prefer environment secrets over rendered TOML.";
      };
      apprise-urls = mkOption {
        type = types.nullOr (types.listOf types.str);
        default = null;
        description = "Apprise notification URLs.";
      };
      email = {
        host = mkOption { type = types.nullOr types.str; default = null; description = "SMTP host."; };
        port = mkOption { type = types.nullOr types.port; default = null; description = "SMTP port."; };
        username = mkOption { type = types.nullOr types.str; default = null; description = "SMTP username."; };
        password = mkOption { type = types.nullOr types.str; default = null; description = "SMTP password; prefer environment secrets over rendered TOML."; };
        from = mkOption { type = types.nullOr types.str; default = null; description = "SMTP sender address."; };
        to = mkOption { type = types.nullOr (types.listOf types.str); default = null; description = "SMTP recipient list."; };
        subject-prefix = mkOption { type = types.nullOr types.str; default = null; description = "SMTP subject prefix."; };
        use-tls = mkOption { type = types.nullOr types.bool; default = null; description = "Use implicit TLS for SMTP."; };
        use-starttls = mkOption { type = types.nullOr types.bool; default = null; description = "Use STARTTLS for SMTP."; };
        max-per-hour = mkOption { type = types.nullOr types.ints.unsigned; default = null; description = "Max notification emails per hour."; };
      };
    };

    health = {
      check-interval-s = mkOption { type = types.nullOr types.ints.unsigned; default = null; description = "Daemon health-check interval."; };
      check-tiers = mkOption { type = types.nullOr types.str; default = null; description = "Health-check tiers."; };
      fts-auto-restore = mkOption { type = types.nullOr types.bool; default = null; description = "Allow health checks to repair FTS trigger drift."; };
      blob-integrity-sample-size = mkOption { type = types.nullOr types.ints.unsigned; default = null; description = "Blob-integrity sample size."; };
    };

    cost = {
      subscription-plans = mkOption {
        type = types.nullOr (types.listOf (types.attrsOf types.unspecified));
        default = null;
        description = "Subscription plan rows rendered as [[cost.subscription.plans]].";
      };
    };
  };

  # systemd resource/hardening knobs that are meaningful for both
  # system and user units. Default values mirror the previous NixOS
  # module so nothing changes for existing deployments.
  serviceOptions = {
    nice = mkOption {
      type = types.ints.between (-20) 19;
      default = 10;
      description = "systemd Nice value for the polylogued unit.";
    };
    ioWeight = mkOption {
      type = types.ints.between 1 10000;
      default = 100;
      description = "systemd IOWeight for the polylogued unit.";
    };
    memoryHigh = mkOption {
      type = types.nullOr types.str;
      default = "2G";
      description = ''
        systemd MemoryHigh for the polylogued unit. Set to null to
        omit the directive entirely.
      '';
    };
    memoryMax = mkOption {
      type = types.nullOr types.str;
      default = "2G";
      description = ''
        systemd MemoryMax for the polylogued unit. Set to null to
        omit the directive entirely.
      '';
    };
  };

  discoverSourcesOption = mkOption {
    type = types.listOf (types.enum providerNames);
    default = [ ];
    example = [ "claude-code" "codex" "gemini" ];
    description = ''
      Convenience: select one or more known provider names and the
      module will add their canonical source paths to
      ``settings.daemon.watch`` automatically. Use this instead of
      hand-spelling paths like ``$HOME/.claude/projects``.

      Recognized values: ${lib.concatStringsSep ", " providerNames}.
    '';
  };

  configLocationOption = mkOption {
    type = types.enum [ "xdg" "store" ];
    default = "xdg";
    description = ''
      Where to render polylogue.toml.

      - ``xdg`` (default): writes ``$XDG_CONFIG_HOME/polylogue/polylogue.toml``
        via Home Manager's ``xdg.configFile``. The file is a symlink
        into the Nix store but lives at a stable user-editable path;
        polylogue's runtime discovery picks it up without needing
        ``POLYLOGUE_CONFIG``.
      - ``store``: writes only into ``/nix/store`` and points the
        unit at it via ``Environment=POLYLOGUE_CONFIG=...``. Use this
        for system-mode deployments or when no per-user XDG path
        applies.
    '';
  };

  # Render the option tree into the nested attrset that
  # pkgs.formats.toml expects. Mirrors polylogue/config.py's TOML
  # schema (section names use snake_case keys like auth_token,
  # allowed_origins, max_cost_usd, force_plain).
  renderSettings = settings:
    let
      dropNulls = lib.filterAttrs (_: v: v != null);
      maybe = section: rendered:
        if rendered == { } then { } else { ${section} = rendered; };

      archive = maybe "archive" (
        lib.optionalAttrs (settings.archive.root != null) {
          root = settings.archive.root;
        }
      );

      daemon = maybe "daemon" (
        dropNulls {
          host = settings.daemon.host;
          port = settings.daemon.port;
        }
        // lib.optionalAttrs (settings.daemon.debounce-s != null) {
          watch = { debounce_s = settings.daemon.debounce-s; };
        }
        // lib.optionalAttrs (
          settings.daemon-api.host != null
          || settings.daemon-api.port != null
          || settings.daemon-api.auth-token != null
        ) {
          api = dropNulls {
            host = settings.daemon-api.host;
            port = settings.daemon-api.port;
            auth_token = settings.daemon-api.auth-token;
          };
        }
        // lib.optionalAttrs (
          settings.browser-capture.host != null
          || settings.browser-capture.port != null
          || settings.browser-capture.allowed-origins != null
          || settings.browser-capture.allow-remote != null
          || settings.browser-capture.auth-token != null
          || settings.browser-capture.spool-path != null
        ) {
          browser_capture = dropNulls {
            host = settings.browser-capture.host;
            port = settings.browser-capture.port;
            allowed_origins = settings.browser-capture.allowed-origins;
            allow_remote = settings.browser-capture.allow-remote;
            auth_token = settings.browser-capture.auth-token;
            spool_path = settings.browser-capture.spool-path;
          };
        }
      );

      sources = maybe "sources" (dropNulls {
        roots = settings.daemon.watch;
      });

      embedding = maybe "embedding" (dropNulls {
        enabled = settings.embedding.enabled;
        model = settings.embedding.model;
        dimension = settings.embedding.dimension;
        max_cost_usd = settings.embedding.max-cost-usd;
        voyage_api_key = settings.embedding.voyage-api-key;
      });

      observability = maybe "observability" (dropNulls {
        enabled = settings.observability.enabled;
        otlp_max_body_bytes = settings.observability.otlp-max-body-bytes;
      });

      logging = maybe "logging" (dropNulls {
        level = settings.logging.level;
        force_plain = settings.logging.force-plain;
      });

      ui = maybe "ui" (dropNulls {
        theme = settings.ui.theme;
        slow_query_notice_seconds = settings.ui.slow-query-notice-seconds;
      });

      schema = maybe "schema" (dropNulls {
        validation = settings.schema.validation;
      });

      notifications = maybe "notifications" (
        dropNulls {
          backend = settings.notifications.backend;
          webhook_url = settings.notifications.webhook-url;
          webhook_secret = settings.notifications.webhook-secret;
          apprise_urls = settings.notifications.apprise-urls;
        }
        // lib.optionalAttrs (
          settings.notifications.email.host != null
          || settings.notifications.email.port != null
          || settings.notifications.email.username != null
          || settings.notifications.email.password != null
          || settings.notifications.email.from != null
          || settings.notifications.email.to != null
          || settings.notifications.email.subject-prefix != null
          || settings.notifications.email.use-tls != null
          || settings.notifications.email.use-starttls != null
          || settings.notifications.email.max-per-hour != null
        ) {
          email = dropNulls {
            host = settings.notifications.email.host;
            port = settings.notifications.email.port;
            username = settings.notifications.email.username;
            password = settings.notifications.email.password;
            from = settings.notifications.email.from;
            to = settings.notifications.email.to;
            subject_prefix = settings.notifications.email.subject-prefix;
            use_tls = settings.notifications.email.use-tls;
            use_starttls = settings.notifications.email.use-starttls;
            max_per_hour = settings.notifications.email.max-per-hour;
          };
        }
      );

      health = maybe "health" (dropNulls {
        check_interval_s = settings.health.check-interval-s;
        check_tiers = settings.health.check-tiers;
        fts_auto_restore = settings.health.fts-auto-restore;
        blob_integrity_sample_size = settings.health.blob-integrity-sample-size;
      });

      cost = maybe "cost" (
        lib.optionalAttrs (settings.cost.subscription-plans != null) {
          subscription = { plans = settings.cost.subscription-plans; };
        }
      );
    in
    archive // daemon // sources // embedding // observability // logging // ui // schema // notifications // health // cost;

  renderConfigFile = settings:
    (pkgs.formats.toml { }).generate "polylogue.toml" (renderSettings settings);

  # Translate ``discoverSources = [ ... ]`` into a list of watch
  # paths. Returned unchanged when the input is empty so callers can
  # safely concatenate.
  expandDiscoverSources = discoverSources:
    map (name: knownProviderRoots.${name}) discoverSources;

  # Compose the effective ``sources.roots`` list by combining the
  # explicit setting (if any) with the discover-sources expansion.
  effectiveWatch = { settings, discoverSources }:
    let
      explicit = settings.daemon.watch or null;
      discovered = expandDiscoverSources discoverSources;
    in
    if explicit == null && discovered == [ ] then
      null
    else
      (if explicit == null then [ ] else explicit) ++ discovered;

  # systemd Service directive block built from service.* options.
  # Returns an attrset suitable for ``serviceConfig`` (system) or
  # ``Service`` (HM). Memory* keys are omitted entirely when null.
  serviceDirectives = service: lib.filterAttrs (_: v: v != null) {
    Nice = service.nice;
    IOWeight = service.ioWeight;
    MemoryHigh = service.memoryHigh;
    MemoryMax = service.memoryMax;
  };

in
{
  inherit
    settingsOptions
    serviceOptions
    discoverSourcesOption
    configLocationOption
    renderSettings
    renderConfigFile
    expandDiscoverSources
    effectiveWatch
    serviceDirectives
    knownProviderRoots
    providerNames
    ;
}
