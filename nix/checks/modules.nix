{
  pkgs,
  nixpkgs,
  homeManager,
  homeModules,
  nixosModules,
  polylogue-extension,
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
  browserDirectories = [
    "/home/alice/.config/browser one"
    "/home/alice/.config/browser two"
  ];
  homeBrowser =
    (home {
      services.polylogue.browser = {
        enable = true;
        userDataDirectories = browserDirectories;
      };
    }).config;
  browserWrapper = lib.findFirst (
    package: lib.getName package == "polylogue-first"
  ) null homeBrowser.home.packages;
  browserManifests = map (
    directory:
    homeBrowser.home.file."${directory}/NativeMessagingHosts/com.polylogue.browser_capture.json".source
  ) browserDirectories;
  browserFiles = homeBrowser.home.file."${homeBrowser.xdg.dataHome}/polylogue/extension";
  systemBase = (system { }).config;
  systemSetting =
    (system {
      services.polylogue.archives.first.settings.http.listen = "127.0.0.1:8899";
      environment.etc."unrelated".text = "independent activation change";
    }).config;
  systemArchive =
    (system { services.polylogue.archives.first.path = lib.mkForce "/home/alice/moved"; }).config;
  homeUnit = evaluated: evaluated.systemd.user.services.polylogue-first;
  systemUnit = evaluated: evaluated.systemd.units."polylogue-first.service".text;
  services = homeBase.systemd.user.services;
  unitSummary = {
    home = homeUnit homeBase;
    system = systemUnit systemBase;
    archiveServices = builtins.attrNames services;
  };
in
assert lib.assertMsg (
  builtins.attrNames services == [
    "polylogue-first"
    "polylogue-second"
  ]
) "Every configured archive must have its own user service.";
assert lib.assertMsg (
  (homeUnit homeBase).Service.Type == "notify"
  && (homeUnit homeBase).Service.WatchdogSec == 30
  && (homeUnit homeBase).Service.RestartPreventExitStatus == "74 78"
  && (homeUnit homeBase).Service.Restart == "always"
  && lib.hasInfix "polylogue-failure-notice" (homeUnit homeBase).Service.ExecStopPost
) "The service must keep readiness, liveness, refused-schema behavior and its failure notice.";
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
assert lib.assertMsg (
  browserFiles.recursive
  && toString browserFiles.source == "${polylogue-extension}/share/polylogue/extension"
) "The Home Manager link must point at the packaged unpacked extension.";
pkgs.runCommand "native_host_manifest_names_archive_wrapper_and_extension_key_id"
  {
    nativeBuildInputs = [ pkgs.jq ];
    unitSummaryJson = builtins.unsafeDiscardStringContext (builtins.toJSON unitSummary);
  }
  ''
    for manifest in ${lib.escapeShellArgs (map toString browserManifests)}; do
      jq -e --arg path "${browserWrapper}/bin/polylogue-first" \
        --arg id pcfmanpibdomhkfjiginenedmjjfgned \
        '.path == $path and .allowed_origins == ["chrome-extension://" + $id + "/"]' "$manifest"
    done
    printf '%s\n' "$unitSummaryJson" > "$out"
  ''
