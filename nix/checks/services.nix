{
  pkgs,
  homeManager,
  homeModule,
  polylogue,
}:
let
  # Home Manager's activation uses an unquoted HOME.
  # Keep the account path simple and exercise literal paths in Polylogue.
  home = "/home/alice";
  special = "${home}/workspace $USER %x";
  expected = {
    HOME = home;
    XDG_CONFIG_HOME = "${home}/.config";
    XDG_DATA_HOME = "${special}/data";
    XDG_STATE_HOME = "${special}/state";
    XDG_CACHE_HOME = "${special}/cache";
    archive = "${special}/archive";
  };
  expectedFile = pkgs.writeText "polylogue-service-paths.json" (builtins.toJSON expected);
in
pkgs.testers.runNixOSTest {
  name = "polylogue-user-service";
  nodes.machine = {
    imports = [ homeManager.nixosModules.home-manager ];
    users.users.alice = {
      isNormalUser = true;
      uid = 1000;
      linger = true;
      inherit home;
    };
    home-manager = {
      useGlobalPkgs = true;
      useUserPackages = false;
      users.alice = {
        imports = [ homeModule ];
        home.stateVersion = "26.05";
        xdg = {
          dataHome = expected.XDG_DATA_HOME;
          stateHome = expected.XDG_STATE_HOME;
          cacheHome = expected.XDG_CACHE_HOME;
        };
        services.polylogue = {
          enable = true;
          package = polylogue;
          archives.main = {
            path = expected.archive;
            settings.http.listen = "127.0.0.1:0";
          };
        };
      };
    };
    environment.systemPackages = [ pkgs.python3 ];
  };
  testScript = ''
    import json
    machine.start()
    machine.wait_for_unit("multi-user.target")
    machine.wait_for_unit("user@1000.service")
    systemctl = "runuser -u alice -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user"
    machine.wait_until_succeeds(systemctl + " is-active polylogue-main.service")
    pid = int(machine.succeed(systemctl + " show --value --property=MainPID polylogue-main.service"))
    try:
        machine.succeed("runuser -u alice -- ${pkgs.python3}/bin/python3 ${./services.py} ${pkgs.lib.escapeShellArg "${home}/.nix-profile/bin"} " + str(pid) + " ${expectedFile}")
    finally:
        machine.succeed(systemctl + " stop polylogue-main.service")
    machine.fail("test -e /proc/" + str(pid))
    machine.succeed("test $(" + systemctl + " show --value --property=ExecMainStatus polylogue-main.service) = 0")
  '';
}
