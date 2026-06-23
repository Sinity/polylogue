# Polylogue NixOS module (system unit).
#
# Pairs with ``nix/hm-module.nix`` for user-mode deployments. Both
# share their option tree and TOML rendering via ``nix/lib/settings.nix``
# so the polylogue.toml schema cannot drift between deployment modes.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.polylogue;
  inherit (lib) mkEnableOption mkOption types;

  settingsLib = import ./lib/settings.nix { inherit lib pkgs; };

  watch = settingsLib.effectiveWatch {
    settings = cfg.settings;
    discoverSources = cfg.discoverSources;
  };

  effectiveSettings = cfg.settings // {
    daemon = cfg.settings.daemon // { inherit watch; };
  };

  configFile = settingsLib.renderConfigFile effectiveSettings;

  # polylogued run consumes the generated TOML through POLYLOGUE_CONFIG;
  # avoid duplicating settings as CLI flags, especially secret-bearing tokens.

in
{
  options.services.polylogue = {
    enable = mkEnableOption "Polylogue AI session archive daemon";

    package = mkOption {
      type = types.package;
      description = "The polylogue package to use.";
    };

    configPath = mkOption {
      type = types.path;
      default = configFile;
      defaultText = lib.literalMD "generated TOML from `services.polylogue.settings`";
      description = ''
        Path to polylogue.toml. Generated from ``settings`` by default
        and passed to the daemon via ``POLYLOGUE_CONFIG``.
      '';
    };

    discoverSources = settingsLib.discoverSourcesOption;

    settings = settingsLib.settingsOptions;

    service = settingsLib.serviceOptions;

    extraServiceConfig = mkOption {
      type = types.attrsOf types.unspecified;
      default = { };
      example = lib.literalExpression ''
        {
          CPUWeight = 50;
          SystemCallFilter = "@system-service";
        }
      '';
      description = ''
        Extra ``serviceConfig`` keys merged onto the polylogued unit.
        Use this for site-specific hardening or scheduler knobs
        without forking the module.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];

    systemd.services.polylogued = {
      description = "Polylogue daemon (live watcher, browser capture, HTTP API)";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = lib.mkMerge [
        {
          Type = "simple";
          # `polylogued run` enables watch + browser-capture + HTTP API by
          # default. Pass --no-watch / --no-browser-capture / --no-api at
          # the unit level if a deployment needs a narrower component set.
          ExecStart = "${cfg.package}/bin/polylogued run";
          Restart = "on-failure";
          RestartSec = "5s";
          Environment = "POLYLOGUE_CONFIG=${cfg.configPath}";
          StateDirectory = "polylogue";
          CacheDirectory = "polylogue";
          IOSchedulingClass = "idle";
          # Practical hardening: the daemon needs read access to source
          # directories (e.g. ~/.claude, ~/.codex) and read/write to the
          # archive root, so full filesystem lockdowns are deferred to
          # the operator.
          PrivateTmp = true;
          NoNewPrivileges = true;
          RestrictAddressFamilies = "AF_UNIX AF_INET AF_INET6";
        }
        (settingsLib.serviceDirectives cfg.service)
        cfg.extraServiceConfig
      ];
    };
  };
}
