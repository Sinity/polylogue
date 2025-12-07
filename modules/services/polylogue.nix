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
      configHome = "/realm/data/chatlog/config";
      dataHome = "/realm/data/chatlog";
      stateDir = "/realm/data/chatlog/state";
      workingDir = "/realm/data/chatlog";
      paths.inputRoot = "/realm/data/chatlog/inbox";
      paths.outputRoot = "/realm/data/chatlog/markdown";
      ui = {
        html = true;
        theme = "dark";
        collapseThreshold = 25;
      };
      watch = {
        gemini = true;
        codex = true;
        claudeCode = true;
        chatgpt = true;
        claude = true;
      };
    };
  };
}
