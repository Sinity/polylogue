{
  pkgs,
  nixpkgs,
  homeManager,
  homeModules,
  nixosModules,
}:
let
  inherit (pkgs) lib;
  app = pkgs.writeShellScriptBin "polylogue" "exit 0";
  otherApp = pkgs.writeShellScriptBin "polylogue" "exit 1";
  archives = {
    first.path = "/home/alice/first archive";
    second.path = "/home/alice/second";
  };
  home =
    extra:
    homeManager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules = [
        homeModules.polylogue
        {
          home = {
            username = "alice";
            homeDirectory = "/home/alice";
            stateVersion = "26.05";
          };
          services.polylogue = {
            enable = true;
            package = app;
            defaultArchive = "first";
            inherit archives;
          };
        }
        extra
      ];
    };
  system =
    extra:
    nixpkgs.lib.nixosSystem {
      system = pkgs.stdenv.hostPlatform.system;
      modules = [
        nixosModules.polylogue
        {
          system.stateVersion = "26.05";
          users.users.alice = {
            isNormalUser = true;
            uid = 1000;
          };
          services.polylogue = {
            enable = true;
            package = app;
            user = "alice";
            defaultArchive = "first";
            inherit archives;
          };
        }
        extra
      ];
    };
  homeBase = (home { }).config;
  homeSetting =
    (home {
      services.polylogue.archives.first.settings.http.listen = "127.0.0.1:8899";
      home.file."unrelated".text = "independent activation change";
    }).config;
  homeArchive =
    (home { services.polylogue.archives.first.path = lib.mkForce "/home/alice/moved"; }).config;
  homeBinary = (home { services.polylogue.package = lib.mkForce otherApp; }).config;
  homeSpecial = (home { home.homeDirectory = lib.mkForce "/home/alice 50% $USER"; }).config;
  systemBase = (system { }).config;
  systemSetting =
    (system {
      services.polylogue.archives.first.settings.http.listen = "127.0.0.1:8899";
      environment.etc."unrelated".text = "independent activation change";
    }).config;
  systemArchive =
    (system { services.polylogue.archives.first.path = lib.mkForce "/home/alice/moved"; }).config;
  homeUnit = value: value.systemd.user.services.polylogue-first;
  systemUnit = value: value.systemd.units."polylogue-first.service".text;
  services = homeBase.systemd.user.services;
  value = {
    home = homeUnit homeBase;
    system = systemUnit systemBase;
    archiveServices = builtins.attrNames services;
  };
in
assert lib.assertMsg (
  builtins.attrNames services == [
    "polylogue-failed@"
    "polylogue-first"
    "polylogue-second"
  ]
) "Every configured archive must have its own user service.";
assert lib.assertMsg (
  (homeUnit homeBase).Service.Type == "notify"
  && (homeUnit homeBase).Service.WatchdogSec == 30
  && (homeUnit homeBase).Service.RestartPreventExitStatus == 78
) "The service must retain readiness, liveness and refused-schema behavior.";
assert lib.assertMsg (
  !(builtins.hasAttr "MemoryMax" (homeUnit homeBase).Service)
) "The module must not impose a memory cap.";
assert lib.assertMsg (
  homeUnit homeBase == homeUnit homeSetting
) "Home Manager config-only activation must leave unit identity unchanged.";
assert lib.assertMsg (
  homeUnit homeBase == homeUnit homeArchive
) "Changing the archive path must take effect on an explicit restart, not activation.";
assert lib.assertMsg (
  (homeUnit homeBase).Unit.X-Restart-Triggers != (homeUnit homeBinary).Unit.X-Restart-Triggers
) "A new application package must change the restart trigger.";
assert lib.assertMsg (
  systemUnit systemBase == systemUnit systemSetting
) "NixOS config-only activation must leave unit identity unchanged.";
assert lib.assertMsg (
  systemUnit systemBase == systemUnit systemArchive
) "NixOS archive path changes must not trigger activation restarts.";
assert lib.assertMsg (
  (homeUnit homeBase).Service.ExecStart
  == [ ''"/home/alice/.nix-profile/bin/polylogue-first" "daemon"'' ]
) "The user service must use the same stable command as its clients.";
assert lib.assertMsg (
  (homeUnit homeSpecial).Service.ExecStart == [
    ''"/home/alice 50%% $$USER/.nix-profile/bin/polylogue-first" "daemon"''
  ]
) "Systemd must preserve literal spaces, dollars and percent signs in the wrapper path.";
# This artifact inspects unit text; it does not execute the paths it names.
pkgs.writeText "polylogue-module-evaluation.json" (
  builtins.unsafeDiscardStringContext (builtins.toJSON value)
)
