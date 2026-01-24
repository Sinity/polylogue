{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.polylogue-sync;

  # Generate config.json from Nix options
  configJson = builtins.toJSON {
    version = 2;
    archive_root = cfg.archiveRoot;
    render_root = cfg.renderRoot;
    sources = cfg.sources;
  };

  # Generate environment variables from configuration
  envVars = {
    POLYLOGUE_ARCHIVE_ROOT = cfg.archiveRoot;
    POLYLOGUE_RENDER_ROOT = cfg.renderRoot;
    XDG_CONFIG_HOME = "/var/lib/polylogue/.config";
    XDG_DATA_HOME = "/var/lib/polylogue/.local/share";
    XDG_STATE_HOME = "/var/lib/polylogue/.local/state";
  } // optionalAttrs (cfg.qdrantUrl != null) {
    POLYLOGUE_QDRANT_URL = cfg.qdrantUrl;
  } // optionalAttrs (cfg.voyageApiKey != null) {
    POLYLOGUE_VOYAGE_API_KEY = cfg.voyageApiKey;
  } // cfg.extraEnv;

in
{
  options.services.polylogue-sync = {
    enable = mkEnableOption "Polylogue automatic sync service";

    package = mkOption {
      type = types.package;
      default = pkgs.polylogue or (throw "polylogue package not found - import polylogue flake or set package explicitly");
      description = "Polylogue package to use";
    };

    user = mkOption {
      type = types.str;
      default = "polylogue";
      description = "User under which polylogue-sync runs";
    };

    group = mkOption {
      type = types.str;
      default = "polylogue";
      description = "Group under which polylogue-sync runs";
    };

    archiveRoot = mkOption {
      type = types.str;
      example = "/realm/data/exports/chatlog/archive";
      description = "Directory where conversation archives are stored";
    };

    renderRoot = mkOption {
      type = types.str;
      example = "/realm/data/exports/chatlog/processed/markdown";
      description = "Directory where rendered outputs are stored";
    };

    sources = mkOption {
      type = types.listOf (types.submodule {
        options = {
          name = mkOption {
            type = types.str;
            description = "Source name (e.g., 'chatgpt', 'claude')";
          };
          path = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Filesystem path to source directory";
          };
          folder = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Google Drive folder name (for Drive API sources)";
          };
        };
      });
      default = [];
      example = literalExpression ''
        [
          { name = "chatgpt"; path = "/realm/data/exports/chatlog/raw/chatgpt"; }
          { name = "claude-code"; path = "''${config.users.users.myuser.home}/.config/claude/projects"; }
          { name = "gemini"; folder = "Google AI Studio"; }
        ]
      '';
      description = "List of sources to sync from";
    };

    syncInterval = mkOption {
      type = types.str;
      default = "hourly";
      example = "*:0/15";
      description = ''
        Systemd timer interval for periodic full syncs.
        Examples: "hourly", "daily", "*:0/15" (every 15 min), "daily *:00:00"
      '';
    };

    watchPaths = mkOption {
      type = types.listOf types.str;
      default = [];
      example = [ "/realm/data/exports/chatlog/raw" ];
      description = ''
        Paths to watch for changes. When files are added/modified,
        polylogue sync is triggered automatically (with 30s debounce).
      '';
    };

    qdrantUrl = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "http://localhost:6333";
      description = "Qdrant server URL for vector search indexing";
    };

    voyageApiKey = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Voyage AI API key for embeddings (use agenix/sops-nix for secrets)";
    };

    extraEnv = mkOption {
      type = types.attrsOf types.str;
      default = {};
      example = {
        POLYLOGUE_DRIVE_RETRIES = "5";
        POLYLOGUE_FORCE_PLAIN = "1";
      };
      description = "Additional environment variables for polylogue";
    };
  };

  config = mkIf cfg.enable {
    # Create user/group
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      description = "Polylogue sync service user";
      home = "/var/lib/polylogue";
      createHome = true;
    };

    users.groups.${cfg.group} = {};

    # Generate config.json via activation script
    system.activationScripts.polylogue-config = stringAfter [ "users" "groups" ] ''
      install -d -m 0700 -o ${cfg.user} -g ${cfg.group} /var/lib/polylogue/.config/polylogue
      cat > /var/lib/polylogue/.config/polylogue/config.json.tmp <<'EOF'
      ${configJson}
      EOF
      mv /var/lib/polylogue/.config/polylogue/config.json{.tmp,}
      chown ${cfg.user}:${cfg.group} /var/lib/polylogue/.config/polylogue/config.json
    '';

    # Periodic sync timer
    systemd.timers.polylogue-sync = {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.syncInterval;
        Persistent = true;
      };
    };

    # Main sync service (triggered by timer)
    systemd.services.polylogue-sync = {
      description = "Polylogue conversation archive sync";
      after = [ "network.target" ];

      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/polylogue sync";
        Environment = mapAttrsToList (name: value: "${name}=${value}") envVars;

        # Hardening
        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictRealtime = true;
        LockPersonality = true;
        ReadWritePaths = [
          cfg.archiveRoot
          cfg.renderRoot
          "/var/lib/polylogue"
        ] ++ cfg.watchPaths;
      };
    };

    # File watcher service (if watchPaths configured)
    systemd.services.polylogue-watch = mkIf (cfg.watchPaths != []) {
      description = "Polylogue file watcher";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        Restart = "on-failure";
        RestartSec = "10s";

        # Use inotifywait to watch for file changes
        ExecStart = pkgs.writeShellScript "polylogue-watch" ''
          #!/bin/sh
          set -eu

          WATCH_PATHS="${concatStringsSep " " cfg.watchPaths}"

          echo "Watching paths: $WATCH_PATHS"

          ${pkgs.inotify-tools}/bin/inotifywait \
            --monitor \
            --recursive \
            --event create,modify,move \
            --format '%w%f' \
            $WATCH_PATHS |
          while IFS= read -r changed_file; do
            # Only trigger on relevant file types
            case "$changed_file" in
              *.json|*.jsonl|*.zip)
                echo "Detected change: $changed_file"
                echo "Triggering sync..."
                systemctl start polylogue-sync.service
                # Debounce: wait 30s before processing more changes
                sleep 30
                ;;
            esac
          done
        '';

        Environment = mapAttrsToList (name: value: "${name}=${value}") envVars;

        # Hardening
        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictRealtime = true;
        LockPersonality = true;
        ReadWritePaths = [
          cfg.archiveRoot
          cfg.renderRoot
          "/var/lib/polylogue"
        ] ++ cfg.watchPaths;
      };
    };
  };
}
