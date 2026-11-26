# Home Manager module: one user service and one command wrapper per archive.
{ packageFor }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.polylogue;
  systemdUtils = import (pkgs.path + "/nixos/lib/utils.nix") { inherit config lib pkgs; };
  deployment = import ./lib/deployment.nix { inherit lib pkgs packageFor; };
  selected = deployment.defaultArchive cfg;
  wrappers = lib.mapAttrs (
    _: archive:
    deployment.wrapper {
      name = archive.command;
      package = cfg.package;
      inherit archive;
      home = config.home.homeDirectory;
      configHome = config.xdg.configHome;
      dataHome = config.xdg.dataHome;
      stateHome = config.xdg.stateHome;
      cacheHome = config.xdg.cacheHome;
      runtimeDir = null;
    }
  ) cfg.archives;
in
{
  options.services.polylogue = deployment.options;
  config = lib.mkIf cfg.enable {
    assertions = deployment.assertions cfg;
    home.packages =
      builtins.attrValues wrappers
      ++ lib.optional (selected != null && builtins.hasAttr selected wrappers) (
        deployment.aliases { inherit cfg selected wrappers; }
      );
    systemd.user.services = lib.mapAttrs' (
      _: archive:
      lib.nameValuePair archive.command {
        Unit = {
          Description = "Polylogue archive ${archive.command}";
          X-Restart-Triggers = [ cfg.package ];
        };
        Service = deployment.serviceConfig cfg // {
          # A profile path keeps TOML/wrapper changes out of unit identity.
          ExecStart = systemdUtils.escapeSystemdExecArgs [
            "${config.home.profileDirectory}/bin/${archive.command}"
            "daemon"
          ];
        };
        Install.WantedBy = lib.optional cfg.autoStart "default.target";
      }
    ) cfg.archives;
  };
}
