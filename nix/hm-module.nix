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
  browserHost = deployment.browserHostManifest {
    extension = cfg.browser.package;
    program = "${wrappers.${selected}}/bin/${cfg.archives.${selected}.command}";
  };
in
{
  options.services.polylogue = deployment.options // {
    browser = {
      enable = lib.mkEnableOption "the Polylogue browser extension";
      package = lib.mkOption {
        type = lib.types.package;
        default = (packageFor pkgs.stdenv.hostPlatform.system).polylogue-extension;
        defaultText = lib.literalExpression "self.packages.<system>.polylogue-extension";
        description = "Extension package from the same build as the application.";
      };
      userDataDirectories = lib.mkOption {
        type = lib.types.listOf (lib.types.strMatching "/.*");
        default = [ "${config.xdg.configHome}/google-chrome" ];
        defaultText = lib.literalExpression ''[ "''${config.xdg.configHome}/google-chrome" ]'';
        description = "Absolute Chrome user-data directories in the home. Each receives the native messaging manifest.";
      };
    };
  };
  config = lib.mkIf cfg.enable {
    assertions = deployment.assertions cfg;
    home.packages =
      builtins.attrValues wrappers
      ++ lib.optional (selected != null && builtins.hasAttr selected wrappers) (
        deployment.aliases { inherit cfg selected wrappers; }
      );
    home.file =
      lib.mkIf (cfg.browser.enable && selected != null && builtins.hasAttr selected wrappers)
        (
          lib.genAttrs
            (map (
              directory: "${directory}/NativeMessagingHosts/com.polylogue.browser_capture.json"
            ) cfg.browser.userDataDirectories)
            (_: {
              source = browserHost;
            })
          // {
            "${config.xdg.dataHome}/polylogue/extension" = {
              # Chrome records the resolved directory on install. Keep the
              # directory itself stable and link the files into it.
              recursive = true;
              source = "${cfg.browser.package}/share/polylogue/extension";
            };
          }
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
