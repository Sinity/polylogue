{ config, lib, pkgs, ... }:

let
  cfg = config.services.polylogue;
  inherit (lib) mkEnableOption mkOption types mdDoc;

  tomlFormat = pkgs.formats.toml { };

  polylogueToml = {
    archive = lib.optionalAttrs (cfg.settings.archive.root != null) {
      root = cfg.settings.archive.root;
    };
    daemon = lib.optionalAttrs (cfg.settings.daemon != { }) (
      lib.filterAttrs (_: v: v != null) {
        host = cfg.settings.daemon.host;
        port = cfg.settings.daemon.port;
        watch = cfg.settings.daemon.watch;
      }
    );
    daemon.api = lib.optionalAttrs (cfg.settings.daemon-api != { }) (
      lib.filterAttrs (_: v: v != null) {
        host = cfg.settings.daemon-api.host;
        port = cfg.settings.daemon-api.port;
        auth_token = cfg.settings.daemon-api.auth-token;
      }
    );
    daemon.browser-capture = lib.optionalAttrs (cfg.settings.browser-capture != { }) (
      lib.filterAttrs (_: v: v != null) {
        port = cfg.settings.browser-capture.port;
        allowed_origins = cfg.settings.browser-capture.allowed-origins;
      }
    );
    embedding = lib.optionalAttrs (cfg.settings.embedding != { }) (
      lib.filterAttrs (_: v: v != null) {
        enabled = cfg.settings.embedding.enabled;
        model = cfg.settings.embedding.model;
        dimension = cfg.settings.embedding.dimension;
        max_cost_usd = cfg.settings.embedding.max-cost-usd;
      }
    );
    hooks = lib.optionalAttrs (cfg.settings.hooks != { }) {
      enabled = cfg.settings.hooks.enabled;
    };
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

      hooks = {
        enabled = mkOption {
          type = types.nullOr types.bool;
          default = null;
          description = "Enable hook-based session capture.";
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
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];

    systemd.services.polylogued = {
      description = "Polylogue daemon";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        ExecStart = "${cfg.package}/bin/polylogued run";
        Restart = "on-failure";
        Environment = "POLYLOGUE_CONFIG=${cfg.configPath}";
        StateDirectory = "polylogue";
        CacheDirectory = "polylogue";
      };
    };
  };
}
