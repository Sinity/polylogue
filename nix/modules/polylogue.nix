{ self }:
{ config, lib, pkgs, ... }:
let
  inherit (lib) mkEnableOption mkOption mkIf types optionalAttrs;

  cfg = config.services.polylogue;

  packagesForSystem = lib.attrByPath [pkgs.system] (self.packages or {}) {};
  defaultPackage = packagesForSystem.polylogue or (
    throw "polylogue package not available for system ${pkgs.system}; make sure self.packages.${pkgs.system}.polylogue is defined"
  );

  configPath = cfg.configHome + "/config.json";

  tessdataDir = "${cfg.ocr.tessdataPackage}/share/tessdata";
  driveCredentialsPath = if cfg.drive.credentialsPath != null
    then cfg.drive.credentialsPath
    else cfg.configHome + "/credentials.json";
  driveTokenPath = if cfg.drive.tokenPath != null
    then cfg.drive.tokenPath
    else cfg.configHome + "/token.json";

  envVars = cfg.environment // {
    XDG_CONFIG_HOME = toString cfg.configHome;
    XDG_DATA_HOME = toString cfg.dataHome;
    XDG_STATE_HOME = toString cfg.stateDir;
    POLYLOGUE_CONFIG = toString configPath;
    POLYLOGUE_FORCE_PLAIN = "1";
    POLYLOGUE_DECLARATIVE = "1";
    TESSDATA_PREFIX = tessdataDir;
    POLYLOGUE_DRIVE_RETRIES = toString cfg.drive.retries;
    POLYLOGUE_DRIVE_RETRY_BASE = toString cfg.drive.retryBase;
  }
  // optionalAttrs (cfg.drive.credentialsPath != null) { POLYLOGUE_CREDENTIAL_PATH = toString cfg.drive.credentialsPath; }
  // optionalAttrs (cfg.drive.tokenPath != null) { POLYLOGUE_TOKEN_PATH = toString cfg.drive.tokenPath; };

  missingOcrLangs = lib.filter (lang: !(builtins.pathExists "${tessdataDir}/${lang}.traineddata")) cfg.ocr.languages;

  localWatchSources = []
    ++ lib.optional cfg.watch.chatgpt { name = "chatgpt"; path = inboxPaths.chatgpt; }
    ++ lib.optional cfg.watch.claude { name = "claude"; path = inboxPaths.claude; }
    ++ lib.optional cfg.watch.codex { name = "codex"; path = inboxDir + "/codex"; }
    ++ lib.optional cfg.watch.claudeCode { name = "claude-code"; path = inboxDir + "/claude-code"; };

  driveSources = lib.optional cfg.watch.gemini {
    name = "gemini";
    folder = cfg.drive.folderName;
  };

  defaultSources = [
    {
      name = "inbox";
      path = inboxDir;
    }
  ];

  resolvedSources = if cfg.sources != [] then cfg.sources
    else if localWatchSources != [] then localWatchSources
    else defaultSources;

  sourcesForConfig = resolvedSources ++ driveSources;

  sourceToJson = source:
    { name = source.name; }
    // optionalAttrs (source.path != null) { path = toString source.path; }
    // optionalAttrs (source.folder != null) { folder = source.folder; };

  configJson = builtins.toJSON {
    version = 2;
    archive_root = toString cfg.paths.outputRoot;
    sources = map sourceToJson sourcesForConfig;
  };

  outputDir = cfg.paths.outputRoot;
  inboxDir = cfg.paths.inputRoot;

  providerPaths = {
    gemini = outputDir + "/gemini";
    codex = outputDir + "/codex";
    claudeCode = outputDir + "/claude-code";
    chatgpt = outputDir + "/chatgpt";
    claude = outputDir + "/claude";
  };

  inboxPaths = {
    chatgpt = inboxDir + "/chatgpt";
    claude = inboxDir + "/claude";
  };

  mkWatchService = name: args:
    let
      cliArgs = args ++ lib.optionals (!cfg.ocr.enable) [ "--no-attachment-ocr" ];
    in {
      systemd.services."polylogue-watch-${name}" = {
        description = "Polylogue ${name} watcher";
        wantedBy = [ "multi-user.target" ];
        after = [ "network-online.target" ];
        path = cfg.helperPackages ++ [ cfg.package ];
        environment = envVars;
        serviceConfig = {
          Type = "simple";
          WorkingDirectory = cfg.workingDir;
          ExecStart = lib.escapeShellArgs ([ "${cfg.package}/bin/polylogue" ] ++ cliArgs);
          Restart = "always";
        } // optionalAttrs (cfg.user != null) { User = cfg.user; };
      };
    };

  runArgs = [ "--plain" "run" ];
  runEnabled = cfg.run.enable || cfg.watch.gemini || cfg.watch.codex || cfg.watch.claudeCode || cfg.watch.chatgpt || cfg.watch.claude;
  runService = mkIf runEnabled {
    systemd.services.polylogue-run = {
      description = "Polylogue ingest/render/index";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      path = cfg.helperPackages ++ [ cfg.package ];
      environment = envVars;
      serviceConfig = {
        Type = "oneshot";
        WorkingDirectory = cfg.workingDir;
        ExecStart = lib.escapeShellArgs ([ "${cfg.package}/bin/polylogue" ] ++ runArgs);
      } // optionalAttrs (cfg.user != null) { User = cfg.user; };
    };
    systemd.timers.polylogue-run = {
      description = "Schedule Polylogue runs";
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnStartupSec = cfg.run.onStartupSec;
        OnUnitActiveSec = cfg.run.onUnitActiveSec;
        Unit = "polylogue-run.service";
      };
    };
  };

  dirs = [ cfg.workingDir cfg.configHome cfg.dataHome cfg.stateDir outputDir inboxDir ]
    ++ lib.attrValues providerPaths
    ++ lib.attrValues inboxPaths;

  tmpfilesRules = lib.map (dir: "d ${dir} 0755 ${cfg.user or "root"} ${cfg.user or "root"} - -") dirs;
in {
  options.services.polylogue = {
    enable = mkEnableOption "Polylogue watch services";

    package = mkOption {
      type = types.package;
      default = defaultPackage;
      description = "Polylogue package providing the CLI binary.";
    };

    user = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Run services as this user (defaults to root).";
    };

    workingDir = mkOption {
      type = types.path;
      default = "/var/lib/polylogue";
      description = "Working directory for watch services.";
    };

    configHome = mkOption {
      type = types.path;
      default = "/var/lib/polylogue/config";
      description = "Directory exported as XDG_CONFIG_HOME.";
    };

    dataHome = mkOption {
      type = types.path;
      default = "/var/lib/polylogue";
      description = "Directory exported as XDG_DATA_HOME and base for inbox/output roots.";
    };

    stateDir = mkOption {
      type = types.path;
      default = "/var/lib/polylogue/state";
      description = "Directory exported as XDG_STATE_HOME (runs DB, tokens).";
    };

    paths = {
      inputRoot = mkOption {
        type = types.path;
        default = "/var/lib/polylogue/inbox";
        description = "Root inbox for provider inputs (chatgpt/claude subdirs are auto-watched).";
      };
      outputRoot = mkOption {
        type = types.path;
        default = "/var/lib/polylogue/archive";
        description = "Root archive for provider outputs (subdirs are fixed per provider).";
      };
    };

    ui = {
      collapseThreshold = mkOption {
        type = types.int;
        default = 25;
        description = "Default collapse threshold.";
      };
      html = mkOption {
        type = types.bool;
        default = true;
        description = "Enable HTML output by default.";
      };
      theme = mkOption {
        type = types.enum [ "light" "dark" ];
        default = "dark";
        description = "Default HTML theme.";
      };
    };

    index = {
      backend = mkOption {
        type = types.enum [ "sqlite" "qdrant" "none" ];
        default = "sqlite";
        description = "Index backend.";
      };
      qdrant = {
        url = mkOption { type = types.nullOr types.str; default = null; }; 
        apiKey = mkOption { type = types.nullOr types.str; default = null; };
        collection = mkOption { type = types.str; default = "polylogue"; };
        vectorSize = mkOption { type = types.nullOr types.int; default = null; };
      };
    };

    watch = {
      gemini = mkOption { type = types.bool; default = false; description = "Enable Drive/Gemini watcher."; };
      codex = mkOption { type = types.bool; default = false; description = "Enable Codex watcher."; };
      claudeCode = mkOption { type = types.bool; default = false; description = "Enable Claude Code watcher."; };
      chatgpt = mkOption { type = types.bool; default = false; description = "Enable ChatGPT exports watcher."; };
      claude = mkOption { type = types.bool; default = false; description = "Enable Claude exports watcher."; };
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Extra environment variables for all services.";
    };

    helperPackages = mkOption {
      type = types.listOf types.package;
      default = with pkgs; [ skim bat glow fd ripgrep jq tesseract ];
      description = "Packages exposed on PATH for Polylogue services (skim/bat/glow pickers plus OCR dependencies).";
    };

    sources = mkOption {
      type = types.listOf (types.submodule {
        options = {
          name = mkOption { type = types.str; };
          path = mkOption { type = types.nullOr types.path; default = null; };
          folder = mkOption { type = types.nullOr types.str; default = null; };
        };
      });
      default = [];
      description = "Explicit source list for config.json (name + path or folder). Overrides defaults when non-empty.";
    };

    run = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = "Enable scheduled polylogue runs via systemd timer.";
      };
      onStartupSec = mkOption {
        type = types.str;
        default = "2min";
        description = "systemd timer OnStartupSec value for polylogue-run.";
      };
      onUnitActiveSec = mkOption {
        type = types.str;
        default = "15min";
        description = "systemd timer OnUnitActiveSec value for polylogue-run.";
      };
    };

    ocr = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable attachment OCR (image text extraction runs by default).";
      };
      languages = mkOption {
        type = types.listOf types.str;
        default = [ "eng" "pol" ];
        description = "Language codes that must be available inside the tessdata directory.";
      };
      tessdataPackage = mkOption {
        type = types.package;
        default = pkgs.tesseract;
        description = "Package providing Tesseract OCR binary + tessdata (share/tessdata is exported via TESSDATA_PREFIX).";
      };
    };

    drive = {
      folderName = mkOption {
        type = types.str;
        default = "Google AI Studio";
        description = "Drive folder name to ingest when Drive source is enabled.";
      };
      credentialsPath = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Path to Drive OAuth client JSON (defaults to configHome/credentials.json).";
      };
      tokenPath = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Path to Drive OAuth token JSON (defaults to configHome/token.json).";
      };
      retries = mkOption {
        type = types.int;
        default = 3;
        description = "Drive retry attempts for sync requests.";
      };
      retryBase = mkOption {
        type = types.float;
        default = 0.5;
        description = "Base delay (seconds) for Drive retry backoff.";
      };
    };
  };

  config = mkIf cfg.enable (
    {
      assertions = lib.optional (cfg.ocr.enable && missingOcrLangs != []) {
        assertion = missingOcrLangs == [];
        message = "Polylogue OCR enabled but tessdata package ${cfg.ocr.tessdataPackage} is missing languages: ${lib.concatStringsSep ", " missingOcrLangs}";
      };

      system.activationScripts.polylogue-config = ''
        install -d -m 0755 ${cfg.configHome}
        cat > ${configPath} <<'EOF'
${configJson}
EOF
      '';

      systemd.tmpfiles.rules = tmpfilesRules;
    }
    // runService
  );
}
