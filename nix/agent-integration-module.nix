# Polylogue native agent-client integration (Home Manager).
#
# This module is intentionally independent from programs.polylogued. It installs
# the package and reconciles user-scoped MCP/guidance files through Polylogue's
# ownership-aware installer; it never starts the daemon or ingests data.
{ config, lib, ... }:

let
  cfg = config.programs.polylogueAgent;
  inherit (lib) mkEnableOption mkIf mkOption types;

  clientArgs = lib.concatMap (client: [ "--client" client ]) cfg.clients;
  archiveArgs = lib.optionals (cfg.archiveRoot != null) [ "--archive-root" cfg.archiveRoot ];
  configArgs = lib.optionals (cfg.configPath != null) [ "--config-path" cfg.configPath ];
  optDownArgs = [
    (if cfg.includeReference then "--reference" else "--no-reference")
    (if cfg.installMcp then "--mcp" else "--no-mcp")
  ];
  replaceArgs = lib.optionals cfg.replaceClients [ "--replace-clients" ];

  installArgs = [
    "${cfg.package}/bin/polylogue"
    "agent"
    "install"
  ] ++ clientArgs ++ [
    "--role" cfg.mcpRole
    "--guidance" cfg.guidance
    "--server-command" "${cfg.package}/bin/polylogue-mcp"
    "--polylogue-command" "${cfg.package}/bin/polylogue"
    "--format" "json"
  ] ++ optDownArgs ++ archiveArgs ++ configArgs ++ replaceArgs;

  installCommand = lib.escapeShellArgs installArgs;
in
{
  options.programs.polylogueAgent = {
    enable = mkEnableOption "Polylogue MCP and standing guidance for native AI agent clients";

    package = mkOption {
      type = types.package;
      description = "Polylogue package whose CLI and MCP server should be installed.";
    };

    clients = mkOption {
      type = types.listOf (types.enum [ "claude-code" "codex" "gemini" "hermes" ]);
      default = [ ];
      example = [ "claude-code" "codex" ];
      description = "Native clients to reconcile. Keep the module enabled while these files are managed.";
    };

    mcpRole = mkOption {
      type = types.enum [ "read" "write" "review" "admin" ];
      default = "read";
      description = "Hard MCP capability role installed into each client command.";
    };

    guidance = mkOption {
      type = types.enum [ "full" "mcp-only" "off" ];
      default = "full";
      description = "Native standing-guidance delivery level.";
    };

    includeReference = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to install the deeper on-demand reference beside native guidance.";
    };

    installMcp = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to install the user-scoped Polylogue MCP entry.";
    };

    archiveRoot = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "/home/alice/.local/share/polylogue/archive";
      description = "Optional explicit POLYLOGUE_ARCHIVE_ROOT passed to each MCP process.";
    };

    configPath = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "/home/alice/.config/polylogue/polylogue.toml";
      description = "Optional explicit POLYLOGUE_CONFIG passed to each MCP process.";
    };

    replaceClients = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Reconcile the selected set and remove exact owned operations for clients
        no longer listed. Drifted native content is retained for operator review.
      '';
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.clients != [ ];
        message = "programs.polylogueAgent.clients must contain at least one client when enabled";
      }
    ];

    home.packages = [ cfg.package ];

    home.activation.polylogueAgentIntegration = lib.hm.dag.entryAfter [ "writeBoundary" ] ''
      run ${installCommand}
    '';
  };
}
