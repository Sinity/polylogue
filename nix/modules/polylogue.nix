{ self }:
{ config, lib, pkgs, ... }:
let
  inherit (lib) mkEnableOption mkOption mkIf mkMerge optionalAttrs types attrNames;

  cfg = config.services.polylogue;
  targetData = builtins.fromJSON (builtins.readFile ../../polylogue/automation_targets.json);

  packagesForSystem = lib.attrByPath [pkgs.system] (self.packages or {}) {};
  defaultPackage = packagesForSystem.polylogue or (
    throw "polylogue package not available for system ${pkgs.system}; make sure self.packages.${pkgs.system}.polylogue is defined"
  );

  targetOption = name: types.submodule ({ ... }:
    let meta = targetData.${name}; in {
      options = {
        enable = mkOption {
          type = types.bool;
          default = false;
          description = "Enable automation for ${meta.description}.";
        };
        workingDir = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = "Working directory for ${meta.description}.";
        };
        outputDir = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = "Output directory passed as --out.";
        };
        extraArgs = mkOption {
          type = types.listOf types.str;
          default = [];
          description = "Additional arguments appended to the Polylogue command.";
        };
        collapseThreshold = mkOption {
          type = types.nullOr types.int;
          default = null;
          description = "Override collapse threshold passed to the CLI.";
        };
        html = mkOption {
          type = types.bool;
          default = false;
          description = "Enable HTML output (--html).";
        };
        timer = mkOption {
          type = types.submodule ({ ... }: {
            options = {
              interval = mkOption {
                type = types.str;
                default = "15m";
                description = "systemd OnUnitActiveSec value.";
              };
              bootDelay = mkOption {
                type = types.str;
                default = "2m";
                description = "systemd OnBootSec value.";
              };
            };
          });
          default = {};
          description = "Timer configuration for the automation target.";
        };
        environment = mkOption {
          type = types.attrsOf types.str;
          default = {};
          description = "Extra environment variables for this target.";
        };
        user = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Run this target as a specific user (overrides services.polylogue.user).";
        };
      };
    }
  );

  targetsOption = types.attrsOf (types.submodule ({ name, ... }:
    if targetData ? ${name}
    then targetOption name
    else throw "Unknown Polylogue automation target '${name}'"
  ));

  allTargets = cfg.targets;
  activeTargets = lib.filterAttrs (_: t: t.enable) allTargets;

  mkServiceFor = name: targetCfg:
    let
      meta = targetData.${name};
      targetUser = targetCfg.user or cfg.user;
      defaults = meta.defaults or {};
      defaultExtra = lib.attrByPath ["extraArgs"] [] defaults;
      defaultCollapse = lib.attrByPath ["collapseThreshold"] null defaults;
      defaultHtml = lib.attrByPath ["html"] false defaults;
      defaultOut = lib.attrByPath ["outputDir"] null defaults;
      defaultWorking = lib.attrByPath ["workingDir"] null defaults;
      workingDir = toString (targetCfg.workingDir or defaultWorking or cfg.workingDir);
      baseArgs = meta.command ++ defaultExtra ++ targetCfg.extraArgs;
      collapseValue = targetCfg.collapseThreshold or defaultCollapse;
      argsWithCollapse = if collapseValue != null && !(lib.elem "--collapse-threshold" baseArgs)
        then baseArgs ++ ["--collapse-threshold", toString collapseValue]
        else baseArgs;
      htmlFlag = if targetCfg.html then true else defaultHtml;
      argsWithHtml = if htmlFlag && !(lib.elem "--html" argsWithCollapse)
        then argsWithCollapse ++ ["--html"]
        else argsWithCollapse;
      resolvedOut = targetCfg.outputDir or defaultOut;
      args = if resolvedOut != null && !(lib.elem "--out" argsWithHtml)
        then argsWithHtml ++ ["--out", toString resolvedOut]
        else argsWithHtml;
      envVars = cfg.environment // targetCfg.environment // {
        XDG_STATE_HOME = toString cfg.stateDir;
        POLYLOGUE_FORCE_PLAIN = "1";
      };
      execStart = lib.escapeShellArgs (["${cfg.package}/bin/polylogue"] ++ args);
      preStart = lib.concatStringsSep "\n" (lib.unique (
        ["mkdir -p ${workingDir}" "mkdir -p ${toString cfg.stateDir}"]
        ++ lib.optionals (resolvedOut != null) ["mkdir -p ${toString resolvedOut}"]
      ));
    in {
      systemd.services.${meta.name} = {
        description = meta.description;
        serviceConfig = {
          Type = "oneshot";
          WorkingDirectory = workingDir;
          ExecStart = execStart;
        } // optionalAttrs (targetUser != null) {
          User = targetUser;
        };
        environment = envVars;
        preStart = preStart;
      };
      systemd.timers.${meta.name} = {
        description = meta.description + " timer";
        wantedBy = [ "timers.target" ];
        partOf = [ meta.name ];
        timerConfig = {
          OnBootSec = targetCfg.timer.bootDelay;
          OnUnitActiveSec = targetCfg.timer.interval;
          Persistent = true;
        };
      };
    };

  tmpfileDirs = lib.unique (
    [ cfg.workingDir cfg.stateDir ]
    ++ (lib.mapAttrsToList (n: t:
      let
        meta = targetData.${n};
        defaults = meta.defaults or {};
      in
        t.outputDir or defaults.outputDir or null
    ) cfg.targets)
  );

  tmpfilesRules = lib.map (dir:
    let owner = cfg.user or "root"; in
    "d ${toString dir} 0755 ${owner} ${owner} - -"
  ) (lib.filter (dir: dir != null) tmpfileDirs);

in {
  options.services.polylogue = {
    enable = mkEnableOption "Polylogue automation timers";

    package = mkOption {
      type = types.package;
      default = defaultPackage;
      description = "Polylogue package providing the CLI binary.";
    };

    user = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Run automation systemd units as this user.";
    };

    workingDir = mkOption {
      type = types.path;
      default = "/var/lib/polylogue";
      description = "Default working directory for automation jobs.";
    };

    stateDir = mkOption {
      type = types.path;
      default = "/var/lib/polylogue/state";
      description = "Directory that will be exported as XDG_STATE_HOME.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Environment variables applied to all targets.";
    };

    targets = mkOption {
      type = targetsOption;
      default = lib.genAttrs (attrNames targetData) (_: {});
      description = "Automation target configuration keyed by target name.";
      example = {
        codex = {
          enable = true;
          outputDir = "/var/lib/polylogue/codex";
          timer.interval = "10m";
        };
        "drive-sync" = {
          enable = true;
          outputDir = "/var/lib/polylogue/gemini-sync";
          extraArgs = [ "--folder-name" "AI Studio" ];
        };
      };
    };
  };

  config = mkIf cfg.enable (
    mkMerge (
      [
        mkMerge (lib.mapAttrsToList mkServiceFor activeTargets)
        { systemd.tmpfiles.rules = tmpfilesRules; }
      ]
    )
  );
}
