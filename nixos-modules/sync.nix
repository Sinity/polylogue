{ config, lib, pkgs, ... }:

let
  cfg = config.services.polylogue-sync;
in
{
  options.services.polylogue-sync = {
    enable = lib.mkEnableOption "Polylogue run --watch service";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.polylogue;
      description = "Polylogue package to use";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.user.services.polylogue-sync = {
      description = "Polylogue continuous run";
      wantedBy = [ "default.target" ];
      after = [ "network-online.target" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/polylogue run --watch";
        Restart = "on-failure";
        RestartSec = "30s";
      };
    };
  };
}
