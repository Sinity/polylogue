# Shared module interface and archive wrapper construction.
{
  lib,
  pkgs,
  packageFor,
}:
let
  inherit (lib) mkOption types;
  settings = import ./settings.nix { inherit lib pkgs; };
  commandName = types.strMatching "[A-Za-z0-9][A-Za-z0-9._+-]*";
  archiveType = types.submodule (
    { name, ... }: {
      imports = settings.archiveType.getSubModules;
      options.command = mkOption {
        type = commandName;
        default = "polylogue-${name}";
        description = "Installed command for this archive. Use a unique shell command name.";
      };
    }
  );
in
{
  options = {
    enable = lib.mkEnableOption "Polylogue archives";
    package = mkOption {
      type = types.package;
      default = (packageFor pkgs.stdenv.hostPlatform.system).polylogue;
      defaultText = lib.literalExpression "self.packages.<system>.polylogue";
      description = "Application package shared by the daemon and every client wrapper.";
    };
    archives = mkOption {
      type = types.attrsOf archiveType;
      default = { };
      description = "Named archives. Each archive runs as a separate service.";
    };
    defaultArchive = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Archive used by polylogue, plg and plog. The only archive is selected automatically.";
    };
    autoStart = mkOption {
      type = types.bool;
      default = true;
      description = "Start archive services with the session or system.";
    };
    watchdogSec = mkOption {
      type = types.ints.positive;
      default = 30;
      description = "Service watchdog interval in seconds.";
    };
    cpuWeight = mkOption {
      type = types.ints.between 1 10000;
      default = 20;
      description = "Relative CPU weight. Interactive work normally has weight 100.";
    };
    ioWeight = mkOption {
      type = types.ints.between 1 10000;
      default = 20;
      description = "Relative I/O weight. No memory limit is imposed.";
    };
  };
  defaultArchive =
    cfg:
    if cfg.defaultArchive != null then
      cfg.defaultArchive
    else if builtins.length (builtins.attrNames cfg.archives) == 1 then
      builtins.head (builtins.attrNames cfg.archives)
    else
      null;
  assertions =
    cfg:
    let
      names = map (archive: archive.command) (builtins.attrValues cfg.archives);
      selected =
        if cfg.defaultArchive != null then
          cfg.defaultArchive
        else if builtins.length (builtins.attrNames cfg.archives) == 1 then
          builtins.head (builtins.attrNames cfg.archives)
        else
          null;
    in
    [
      {
        assertion = cfg.archives != { };
        message = "Polylogue needs at least one declared archive.";
      }
      {
        assertion = selected != null && builtins.hasAttr selected cfg.archives;
        message = "Select services.polylogue.defaultArchive when declaring several archives.";
      }
      {
        assertion =
          builtins.length names == builtins.length (lib.unique names)
          && !builtins.any (
            name:
            builtins.elem name [
              "polylogue"
              "plg"
              "plog"
            ]
          ) names;
        message = "Polylogue archive commands must be unique and distinct from polylogue, plg and plog.";
      }
    ];
  serviceConfig = cfg: {
    Type = "notify";
    NotifyAccess = "main";
    WatchdogSec = cfg.watchdogSec;
    Restart = "on-failure";
    RestartPreventExitStatus = 78;
    # A stop or restart signals the daemon only: the `polylogue host`
    # processes of hosted sessions, and headless resumed runs, keep their
    # agents running, and the next daemon reattaches to them.
    KillMode = "process";
    CPUWeight = cfg.cpuWeight;
    IOWeight = cfg.ioWeight;
    IOSchedulingClass = "idle";
  };
  wrapper = settings.wrapper;
  aliases =
    {
      cfg,
      selected,
      wrappers,
    }:
    pkgs.runCommand "polylogue-default-commands" { } ''
      mkdir -p "$out/bin"
      ${lib.concatMapStringsSep "\n"
        (name: ''
          ln -s ${lib.escapeShellArg "${wrappers.${selected}}/bin/${cfg.archives.${selected}.command}"} "$out/bin/${name}"
        '')
        [
          "polylogue"
          "plg"
          "plog"
        ]
      }
    '';
}
