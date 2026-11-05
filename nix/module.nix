{ packageFor }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.polylogue;
  deployment = import ./lib/deployment.nix { inherit lib pkgs packageFor; };
  selected = deployment.defaultArchive cfg;
  wrappers = lib.mapAttrs (
    _: archive:
    deployment.wrapper {
      name = archive.command;
      package = cfg.package;
      inherit archive;
      home = cfg.home;
      config = cfg.configHome;
      data = cfg.dataHome;
      state = cfg.stateHome;
      cache = cfg.cacheHome;
      runtime = "/run/${archive.command}";
    }
  ) cfg.archives;
  directoryOption =
    suffix:
    lib.mkOption {
      type = lib.types.strMatching "/.*";
      default = "${cfg.home}/${suffix}";
      description = "Explicit writable XDG directory shared by the daemon and clients.";
    };
in
{
  options.services.polylogue = deployment.options // {
    user = lib.mkOption {
      type = lib.types.str;
      description = "Existing account whose sources and archive the service owns.";
    };
    home = lib.mkOption {
      type = lib.types.strMatching "/.*";
      default = config.users.users.${cfg.user}.home;
      description = "Home directory used for source discovery.";
    };
    dataHome = directoryOption ".local/share";
    configHome = directoryOption ".config";
    stateHome = directoryOption ".local/state";
    cacheHome = directoryOption ".cache";
  };
  config = lib.mkIf cfg.enable {
    assertions = deployment.assertions cfg ++ [
      {
        assertion = builtins.hasAttr cfg.user config.users.users;
        message = "Polylogue's service account must be declared in users.users.";
      }
    ];
    environment.systemPackages =
      builtins.attrValues wrappers
      ++ lib.optional (selected != null && builtins.hasAttr selected wrappers) (
        deployment.aliases { inherit cfg selected wrappers; }
      );
    systemd.services =
      lib.mapAttrs' (
        _: archive:
        lib.nameValuePair archive.command {
          description = "Polylogue archive ${archive.command}";
          wantedBy = lib.optional cfg.autoStart "multi-user.target";
          after = [ "network.target" ];
          restartTriggers = [ cfg.package ];
          onFailure = [ "polylogue-failed@%n.service" ];
          serviceConfig = deployment.serviceConfig cfg // {
            User = cfg.user;
            WorkingDirectory = builtins.toJSON (lib.replaceStrings [ "%" ] [ "%%" ] cfg.home);
            RuntimeDirectory = archive.command;
            RuntimeDirectoryMode = "0700";
            ExecStart = "/run/current-system/sw/bin/${archive.command} daemon";
          };
        }
      ) cfg.archives
      // {
        "polylogue-failed@" = {
          description = "Notify about a failed Polylogue service";
          serviceConfig = {
            Type = "oneshot";
            User = cfg.user;
            Environment = "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/%U/bus";
            ExecStart = "${lib.getExe pkgs.libnotify} --urgency=critical 'Polylogue stopped' 'Service %i failed; inspect its journal.'";
          };
        };
      };
  };
}
