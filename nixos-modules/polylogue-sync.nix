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
      default = pkgs.polylogue;
      defaultText = literalExpression "pkgs.polylogue";
      description = "Polylogue package to use";
    };

    syncInterval = mkOption {
      type = types.str;
      default = "hourly";
      example = "*:0/15";
      description = ''
        Systemd timer interval for periodic syncs.
        Examples: "hourly", "daily", "*:0/15" (every 15 min)
      '';
    };

    enableGoogleDrive = mkOption {
      type = types.bool;
      default = false;
      description = "Enable Google Drive (Gemini) ingestion";
    };
  };

  config = mkIf cfg.enable {
    # Periodic sync timer
    systemd.timers.polylogue-sync = {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.syncInterval;
        Persistent = true;
      };
    };

    # Main sync service
    systemd.services.polylogue-sync = {
      description = "Polylogue conversation archive sync";
      after = [ "network.target" ];

      serviceConfig = {
        Type = "oneshot";
        ExecStart = "${cfg.package}/bin/polylogue sync";

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
      };
    };
  };
}
