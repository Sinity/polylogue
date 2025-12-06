{ self }:
{ config, lib, pkgs, ... }:
let
  inherit (lib) mkEnableOption mkOption mkIf mkMerge optionalAttrs types attrNames dirOf filterAttrs mapAttrsToList;

  cfg = config.services.polylogue;
  configFilePath = if cfg.configFile.path != null then cfg.configFile.path else cfg.configHome + "/config.json";
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
      workingDir = toString (
        if targetCfg.workingDir != null then targetCfg.workingDir
        else if defaultWorking != null then defaultWorking
        else cfg.workingDir
      );
      baseArgs = meta.command ++ defaultExtra ++ targetCfg.extraArgs;
      collapseValue = targetCfg.collapseThreshold or defaultCollapse;
      argsWithCollapse = if collapseValue != null && !(lib.elem "--collapse-threshold" baseArgs)
        then baseArgs ++ ["--collapse-threshold" (toString collapseValue)]
        else baseArgs;
      htmlFlag = if targetCfg.html then true else defaultHtml;
      argsWithHtml = if htmlFlag && !(lib.elem "--html" argsWithCollapse)
        then argsWithCollapse ++ ["--html"]
        else argsWithCollapse;
      resolvedOut = targetCfg.outputDir or defaultOut;
      args = if resolvedOut != null && !(lib.elem "--out" argsWithHtml)
        then argsWithHtml ++ ["--out" (toString resolvedOut)]
        else argsWithHtml;
      envVars = baseEnv // targetCfg.environment
        // optionalAttrs cfg.configFile.enable { POLYLOGUE_CONFIG = toString configFilePath; };
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
    [ cfg.workingDir cfg.stateDir cfg.configHome ]
    ++ (mapAttrsToList (_: path: path) mergedOutputDirs)
    ++ (mapAttrsToList (n: t:
      let
        meta = targetData.${n};
        defaults = meta.defaults or {};
      in
        t.outputDir or defaults.outputDir or null
    ) cfg.targets)
    ++ lib.optional (cfg.drive.tokenPath != null) cfg.drive.tokenPath
  );

  tmpfilesRules = lib.map (dir:
    let owner = cfg.user or "root"; in
    "d ${toString dir} 0755 ${owner} ${owner} - -"
  ) (lib.filter (dir: dir != null) tmpfileDirs);

  markdownRoot = cfg.dataHome + "/archive/markdown";
  defaultOutputDirs = {
    render = markdownRoot + "/gemini-render";
    sync_drive = markdownRoot + "/gemini-sync";
    sync_codex = markdownRoot + "/codex";
    sync_claude_code = markdownRoot + "/claude-code";
    import_chatgpt = markdownRoot + "/chatgpt";
    import_claude = markdownRoot + "/claude";
  };

  mergedOutputDirs = defaultOutputDirs // (filterAttrs (_: v: v != null) cfg.configFile.settings.outputDirs);
  defaultsJsonLines = filterAttrs (_: v: v != null) {
    collapse_threshold = cfg.configFile.settings.collapseThreshold;
    html_previews = cfg.configFile.settings.htmlPreviews;
    html_theme = cfg.configFile.settings.htmlTheme;
  };

  configJson = builtins.toJSON {
    defaults = defaultsJsonLines // {
      output_dirs = lib.mapAttrs (_: toString) mergedOutputDirs;
    };
  };

  indexBackend = if cfg.qdrant.enable then "qdrant" else cfg.indexBackend;
  driveEnv = filterAttrs (_: v: v != null) {
    POLYLOGUE_DRIVE_CREDENTIALS = cfg.drive.credentialsPath;
    POLYLOGUE_TOKEN_PATH = cfg.drive.tokenPath;
    POLYLOGUE_RETRIES = if cfg.drive.retries != null then toString cfg.drive.retries else null;
    POLYLOGUE_RETRY_BASE = if cfg.drive.retryBase != null then toString cfg.drive.retryBase else null;
  };
  qdrantEnv = if indexBackend == "qdrant" then filterAttrs (_: v: v != null) {
    POLYLOGUE_INDEX_BACKEND = indexBackend;
    POLYLOGUE_QDRANT_URL = cfg.qdrant.url;
    POLYLOGUE_QDRANT_API_KEY = cfg.qdrant.apiKey;
    POLYLOGUE_QDRANT_COLLECTION = cfg.qdrant.collection;
    POLYLOGUE_QDRANT_VECTOR_SIZE = if cfg.qdrant.vectorSize != null then toString cfg.qdrant.vectorSize else null;
  } else {
    POLYLOGUE_INDEX_BACKEND = indexBackend;
  };
  sessionEnv = filterAttrs (_: v: v != null) {
    POLYLOGUE_CODEX_SESSIONS = cfg.sessionRoots.codexSessions;
    POLYLOGUE_CLAUDE_CODE_PROJECTS = cfg.sessionRoots.claudeCodeProjects;
  };
  baseEnv = cfg.environment // driveEnv // qdrantEnv // sessionEnv // {
    XDG_CONFIG_HOME = toString cfg.configHome;
    XDG_DATA_HOME = toString cfg.dataHome;
    XDG_STATE_HOME = toString cfg.stateDir;
    POLYLOGUE_FORCE_PLAIN = "1";
  };

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

    configHome = mkOption {
      type = types.path;
      default = "/var/lib/polylogue/config";
      description = "Directory exported as XDG_CONFIG_HOME for services.";
    };

    dataHome = mkOption {
      type = types.path;
      default = "/var/lib/polylogue";
      description = "Directory exported as XDG_DATA_HOME for services and output roots.";
    };

    configFile = {
      enable = mkEnableOption "Write a polylogue config.json for services" // { default = true; };
      path = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Location of the generated polylogue config.json (also exported via POLYLOGUE_CONFIG). Defaults to ${config.services.polylogue.configHome}/config.json.";
      };
      settings = {
        collapseThreshold = mkOption {
          type = types.nullOr types.int;
          default = null;
          description = "defaults.collapse_threshold value; null uses application default.";
        };
        htmlPreviews = mkOption {
          type = types.nullOr types.bool;
          default = null;
          description = "defaults.html_previews value; null uses application default.";
        };
        htmlTheme = mkOption {
          type = types.nullOr (types.enum [ "light" "dark" ]);
          default = null;
          description = "defaults.html_theme value; null uses application default.";
        };
        outputDirs = mkOption {
          type = types.attrsOf (types.nullOr types.path);
          default = {};
          description = "Optional overrides for defaults.output_dirs (render/sync_drive/sync_codex/sync_claude_code/import_chatgpt/import_claude).";
        };
      };
    };

    stateDir = mkOption {
      type = types.path;
      default = "/var/lib/polylogue/state";
      description = "Directory that will be exported as XDG_STATE_HOME (runs DB, tokens).";
    };

    indexBackend = mkOption {
      type = types.enum [ "sqlite" "qdrant" "none" ];
      default = "sqlite";
      description = "Index backend to advertise; set to qdrant to send vectors to Qdrant.";
    };

    drive = {
      credentialsPath = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Path to Google OAuth credentials.json (POLYLOGUE_DRIVE_CREDENTIALS).";
      };
      tokenPath = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Path for Drive token.json (POLYLOGUE_TOKEN_PATH).";
      };
      retries = mkOption {
        type = types.nullOr types.int;
        default = null;
        description = "Retry attempts for Drive requests (POLYLOGUE_RETRIES).";
      };
      retryBase = mkOption {
        type = types.nullOr types.number;
        default = null;
        description = "Base delay for Drive retries in seconds (POLYLOGUE_RETRY_BASE).";
      };
    };

    qdrant = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = "Enable Qdrant indexing (forces POLYLOGUE_INDEX_BACKEND=qdrant).";
      };
      url = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Qdrant URL (POLYLOGUE_QDRANT_URL).";
      };
      apiKey = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Qdrant API key (POLYLOGUE_QDRANT_API_KEY); leave null for localhost without auth.";
      };
      collection = mkOption {
        type = types.nullOr types.str;
        default = "polylogue";
        description = "Qdrant collection name (POLYLOGUE_QDRANT_COLLECTION).";
      };
      vectorSize = mkOption {
        type = types.nullOr types.int;
        default = null;
        description = "Qdrant vector size override (POLYLOGUE_QDRANT_VECTOR_SIZE).";
      };
    };

    sessionRoots = {
      codexSessions = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Override POLYLOGUE_CODEX_SESSIONS path for local Codex sessions.";
      };
      claudeCodeProjects = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Override POLYLOGUE_CLAUDE_CODE_PROJECTS path for Claude Code projects.";
      };
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
        (mkIf cfg.configFile.enable {
          system.activationScripts.polylogue-config = lib.stringAfter [ "etc" ] ''
            install -d -m 0755 ${dirOf configFilePath}
            cat > ${configFilePath} <<'EOF'
${configJson}
EOF
          '';
        })
      ]
    )
  );
}
