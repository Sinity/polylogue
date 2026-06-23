# Polylogue Home Manager module (per-user systemd unit).
#
# Pairs with ``nix/module.nix`` for system-mode deployments. Both
# share their option tree and TOML rendering via ``nix/lib/settings.nix``
# so the polylogue.toml schema cannot drift between deployment modes.
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.polylogued;
  inherit (lib) mkEnableOption mkOption types;

  settingsLib = import ./lib/settings.nix { inherit lib pkgs; };

  watch = settingsLib.effectiveWatch {
    settings = cfg.settings;
    discoverSources = cfg.discoverSources;
  };

  effectiveSettings = cfg.settings // {
    daemon = cfg.settings.daemon // { inherit watch; };
  };

  storeConfigFile = settingsLib.renderConfigFile effectiveSettings;

  xdgRelPath = "polylogue/polylogue.toml";

  # When configLocation = "xdg" the polylogued unit reads the user's
  # editable config from $XDG_CONFIG_HOME, which is polylogue's normal
  # discovery path — no POLYLOGUE_CONFIG override needed. When
  # ``store`` the unit gets a /nix/store path passed via the env var.
  useXdg = cfg.configLocation == "xdg";

  # polylogued run consumes the generated TOML directly; keep
  # secret-bearing values out of ExecStart arguments.

in
{
  options.programs.polylogued = {
    enable = mkEnableOption "Polylogue AI session archive daemon (user-mode)";

    package = mkOption {
      type = types.package;
      description = "The polylogue package to use.";
    };

    autoStart = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to start the polylogued user systemd unit at login
        (``WantedBy = default.target``).
      '';
    };

    configLocation = settingsLib.configLocationOption;

    discoverSources = settingsLib.discoverSourcesOption;

    settings = settingsLib.settingsOptions;

    service = settingsLib.serviceOptions;

    extraServiceConfig = mkOption {
      type = types.attrsOf types.unspecified;
      default = { };
      example = lib.literalExpression ''
        {
          Slice = "background.slice";
          CPUWeight = 50;
        }
      '';
      description = ''
        Extra ``Service`` keys merged onto the polylogued user unit.
        Use this for site-specific resource-class wiring (slices,
        cgroup hooks) without forking the module.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ cfg.package ];

    xdg.configFile = lib.mkIf useXdg {
      ${xdgRelPath} = {
        source = storeConfigFile;
      };
    };

    systemd.user.services.polylogued = {
      Unit = {
        Description = "Polylogue daemon (live watcher, browser capture, HTTP API)";
        After = [ "default.target" ];
      };
      Service = lib.mkMerge [
        {
          Type = "simple";
          ExecStart = "${cfg.package}/bin/polylogued run";
          Restart = "on-failure";
          RestartSec = "5s";
        }
        (lib.optionalAttrs (!useXdg) {
          Environment = "POLYLOGUE_CONFIG=${storeConfigFile}";
        })
        (settingsLib.serviceDirectives cfg.service)
        cfg.extraServiceConfig
      ];
      Install = lib.mkIf cfg.autoStart {
        WantedBy = [ "default.target" ];
      };
    };
  };
}
