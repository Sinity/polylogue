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

  envVars = cfg.environment // {
    XDG_CONFIG_HOME = toString cfg.configHome;
    XDG_DATA_HOME = toString cfg.dataHome;
    XDG_STATE_HOME = toString cfg.stateDir;
    POLYLOGUE_FORCE_PLAIN = "1";
  };

  configJson = builtins.toJSON {
    paths = {
      input_root = toString cfg.paths.inputRoot;
      output_root = toString cfg.paths.outputRoot;
    };
    ui = {
      collapse_threshold = cfg.ui.collapseThreshold;
      html = cfg.ui.html;
      theme = cfg.ui.theme;
    };
    index = {
      backend = cfg.index.backend;
      qdrant = {
        url = cfg.index.qdrant.url;
        api_key = cfg.index.qdrant.apiKey;
        collection = cfg.index.qdrant.collection;
        vector_size = cfg.index.qdrant.vectorSize;
      };
    };
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

  mkWatchService = name: args: {
    systemd.services."polylogue-watch-${name}" = {
      description = "Polylogue ${name} watcher";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      environment = envVars;
      serviceConfig = {
        Type = "simple";
        WorkingDirectory = cfg.workingDir;
        ExecStart = lib.escapeShellArgs ([ "${cfg.package}/bin/polylogue" ] ++ args);
        Restart = "always";
      };
    };
  };

  watchServices = lib.mkMerge (
    []
    ++ lib.optional cfg.watch.gemini (mkWatchService "gemini" [ "sync" "drive" "--watch" "--out" providerPaths.gemini ])
    ++ lib.optional cfg.watch.codex (mkWatchService "codex" [ "sync" "codex" "--watch" "--out" providerPaths.codex ])
    ++ lib.optional cfg.watch.claudeCode (mkWatchService "claude-code" [ "sync" "claude-code" "--watch" "--out" providerPaths.claudeCode ])
    ++ lib.optional cfg.watch.chatgpt (mkWatchService "chatgpt" [ "sync" "chatgpt" "--watch" "--base-dir" inboxPaths.chatgpt "--out" providerPaths.chatgpt ])
    ++ lib.optional cfg.watch.claude (mkWatchService "claude" [ "sync" "claude" "--watch" "--base-dir" inboxPaths.claude "--out" providerPaths.claude ])
  );

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
  };

  config = mkIf cfg.enable (
    {
      system.activationScripts.polylogue-config = ''
        install -d -m 0755 ${cfg.configHome}
        cat > ${configPath} <<'EOF'
${configJson}
EOF
      '';

      systemd.tmpfiles.rules = tmpfilesRules;
    }
    // watchServices
  );
}
