{
  pkgs,
  homeManager,
  homeModule,
  polylogue,
}:
let
  home = "/home/alice workspace $USER %x";
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
        services.polylogue = {
          enable = true;
          package = polylogue;
          archives.main = {
            path = "${home}/archive";
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
        machine.succeed("runuser -u alice -- ${pkgs.python3}/bin/python3 ${./services.py} ${pkgs.lib.escapeShellArg "${home}/.nix-profile/bin"} " + str(pid))
    finally:
        machine.succeed(systemctl + " stop polylogue-main.service")
    machine.fail("test -e /proc/" + str(pid))
    machine.succeed("test $(" + systemctl + " show --value --property=ExecMainStatus polylogue-main.service) = 0")
  '';
}
