{ config, lib, pkgs, inputs, ... }:
let
  inherit (lib) mkEnableOption mkIf;
  polyloguePkg = inputs.polylogue.packages.${pkgs.stdenv.hostPlatform.system}.polylogue;
in
{
  options.services.polylogue-sinnix.enable = mkEnableOption "Polylogue watch services (sinnix profile)";

  config = mkIf config.services.polylogue-sinnix.enable {
    imports = [ inputs.polylogue.nixosModules.polylogue ];

    services.polylogue = {
      enable = true;
      package = polyloguePkg;
      configHome = "/realm/data/exports/chatlog/config";
      dataHome = "/realm/data/exports/chatlog";
      stateDir = "/realm/data/exports/chatlog/state";
      workingDir = "/realm/data/exports/chatlog";
      paths.inputRoot = "/realm/data/exports/chatlog/raw/inbox";
      paths.outputRoot = "/realm/data/exports/chatlog/archive";
      paths.renderRoot = "/realm/data/exports/chatlog/processed/markdown";
      ui = {
        html = true;
        theme = "dark";
        collapseThreshold = 25;
      };
      run.enable = true;
      watch = {
        gemini = false;
        codex = false;
        claudeCode = false;
        chatgpt = false;
        claude = false;
      };
      sources = [
        { name = "codex"; path = "/home/sinity/.codex/sessions"; }
        { name = "claude-code"; path = "/home/sinity/.config/claude/projects"; }
        { name = "chatgpt"; path = "/realm/data/exports/chatlog/raw/chatgpt"; }
        { name = "claude"; path = "/realm/data/exports/chatlog/raw/claude"; }
        { name = "inbox"; path = "/realm/data/exports/chatlog/raw/inbox"; }
        { name = "gemini"; folder = "Google AI Studio"; }
      ];
    };
  };
}
