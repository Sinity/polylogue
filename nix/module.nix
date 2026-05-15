{ config, lib, pkgs, ... }:

let
  cfg = config.services.polylogue;
  inherit (lib) mkEnableOption mkOption types mdDoc;

  tomlFormat = pkgs.formats.toml { };

  polylogueToml = {
    archive = lib.optionalAttrs (cfg.settings.archive.root != null) {
      root = cfg.settings.archive.root;
    };
    daemon =
      let
        base = lib.optionalAttrs (cfg.settings.daemon != { }) (
          lib.filterAttrs (_: v: v != null) {
            host = cfg.settings.daemon.host;
            port = cfg.settings.daemon.port;
            watch = cfg.settings.daemon.watch;
          }
        );
        api = lib.optionalAttrs (cfg.settings.daemon-api != { }) {
          api = lib.filterAttrs (_: v: v != null) {
            host = cfg.settings.daemon-api.host;
            port = cfg.settings.daemon-api.port;
            auth_token = cfg.settings.daemon-api.auth-token;
          };
        };
        browser-capture = lib.optionalAttrs (cfg.settings.browser-capture != { }) {
          browser-capture = lib.filterAttrs (_: v: v != null) {
            port = cfg.settings.browser-capture.port;
            allowed_origins = cfg.settings.browser-capture.allowed-origins;
          };
        };
      in
      base // api // browser-capture;
    embedding = lib.optionalAttrs (cfg.settings.embedding != { }) (
      lib.filterAttrs (_: v: v != null) {
        enabled = cfg.settings.embedding.enabled;
        model = cfg.settings.embedding.model;
        dimension = cfg.settings.embedding.dimension;
        max_cost_usd = cfg.settings.embedding.max-cost-usd;
      }
    );
    logging = lib.optionalAttrs (cfg.settings.logging != { }) (
      lib.filterAttrs (_: v: v != null) {
        level = cfg.settings.logging.level;
        force_plain = cfg.settings.logging.force-plain;
      }
    );
  };

  configFile = tomlFormat.generate "polylogue.toml" polylogueToml;

in
{
  options.services.polylogue = {
    enable = mkEnableOption "Polylogue AI conversation archive daemon";

    package = mkOption {
      type = types.package;
      description = "The polylogue package to use.";
    };

    configPath = mkOption {
      type = types.path;
      default = configFile;
      defaultText = "generated TOML from services.polylogue.settings";
      description = "Path to polylogue.toml. Generated from settings by default.";
    };

    settings = {
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
          description = "Daemon listen host (default: 127.0.0.1).";
        };
        port = mkOption {
          type = types.nullOr types.port;
          default = null;
          description = "Daemon listen port (default: 8766).";
        };
        watch = mkOption {
          type = types.nullOr (types.listOf types.str);
          default = null;
          description = "Additional watch roots for live ingestion.";
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
          description = "API listen port.";
        };
        auth-token = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "API Bearer auth token.";
        };
      };

      browser-capture = {
        port = mkOption {
          type = types.nullOr types.port;
          default = null;
          description = "Browser capture receiver port.";
        };
        allowed-origins = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Comma-separated allowed CORS origins.";
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
    };

    service = {
      nice = mkOption {
        type = types.ints.between (-20) 19;
        default = 10;
        description = "systemd Nice value for the Polylogue daemon.";
      };
      ioWeight = mkOption {
        type = types.ints.between 1 10000;
        default = 100;
        description = "systemd IOWeight for the Polylogue daemon.";
      };
      memoryHigh = mkOption {
        type = types.nullOr types.str;
        default = "1G";
        description = "systemd MemoryHigh value for the Polylogue daemon.";
      };
      memoryMax = mkOption {
        type = types.nullOr types.str;
        default = "2G";
        description = "systemd MemoryMax value for the Polylogue daemon.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];

    systemd.services.polylogued = {
      description = "Polylogue daemon (live watcher, browser capture, HTTP API)";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        Type = "simple";
        # `polylogued run` enables watch + browser-capture + HTTP API by default.
        # Pass --no-watch / --no-browser-capture / --no-api at the unit level if
        # a deployment needs a narrower component set.
        ExecStart = "${cfg.package}/bin/polylogued run";
        Restart = "on-failure";
        RestartSec = "5s";
        Environment = "POLYLOGUE_CONFIG=${cfg.configPath}";
        StateDirectory = "polylogue";
        CacheDirectory = "polylogue";
        Nice = cfg.service.nice;
        IOSchedulingClass = "idle";
        IOWeight = cfg.service.ioWeight;
        MemoryHigh = cfg.service.memoryHigh;
        MemoryMax = cfg.service.memoryMax;
        # Practical hardening: the daemon needs read access to source directories
        # (e.g. ~/.claude, ~/.codex) and read/write to the archive root, so full
        # filesystem lockdowns are deferred to the operator. These settings are
        # safe for any deployment.
        PrivateTmp = true;
        NoNewPrivileges = true;
        RestrictAddressFamilies = "AF_UNIX AF_INET AF_INET6";
      };
    };
  };
}
