{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.polylogue-sync;
in
{
  options.services.polylogue-sync = {
    enable = mkEnableOption "Polylogue automatic sync service";

    package = mkOption {
      type = types.package;
      default = pkgs.python313Packages.buildPythonApplication {
        pname = "polylogue";
        version = "0.1.0";
        pyproject = true;
        src = ../.;  # Points to polylogue repo root
        propagatedBuildInputs = with pkgs.python313Packages; [
          # Core dependencies - sync with pyproject.toml
          click
          rich
          pydantic
          dependency-injector
          google-api-python-client
          google-auth-oauthlib
          httpx
        ];
        nativeBuildInputs = with pkgs.python313Packages; [
          setuptools
          wheel
        ];
      };
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

    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      example = {
        POLYLOGUE_ARCHIVE_ROOT = "/realm/data/exports/chatlog";
        POLYLOGUE_QDRANT_URL = "http://localhost:6333";
      };
      description = ''
        Environment variables for polylogue.
        See polylogue documentation for available variables.
      '';
    };

    watchPaths = mkOption {
      type = types.listOf types.str;
      default = [];
      example = [ "/realm/data/exports/chatlog" ];
      description = ''
        Paths to watch for changes. When files are added/modified,
        polylogue sync is triggered automatically.
      '';
    };

    syncInterval = mkOption {
      type = types.str;
      default = "hourly";
      description = ''
        Systemd timer interval for periodic full syncs.
        Examples: "hourly", "daily", "*:0/15" (every 15 min)
      '';
    };
  };

  config = mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      description = "Polylogue sync service user";
      home = "/var/lib/polylogue";
      createHome = true;
    };

    users.groups.${cfg.group} = {};

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
        Environment = mapAttrsToList (name: value: "${name}=${value}") cfg.environment;

        # Hardening
        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/lib/polylogue" ] ++ cfg.watchPaths;
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

        Environment = mapAttrsToList (name: value: "${name}=${value}") cfg.environment;

        # Hardening
        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ "/var/lib/polylogue" ] ++ cfg.watchPaths;
      };
    };
  };
}
