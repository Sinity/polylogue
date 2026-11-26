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
  # A one-commit repository with an origin remote, made at build time: the
  # guest has no git of its own. `root` holds the commit the hook must report.
  gitFixture =
    pkgs.runCommand "polylogue-git-fixture"
      {
        nativeBuildInputs = [ pkgs.gitMinimal ];
        GIT_AUTHOR_NAME = "Morrow";
        GIT_AUTHOR_EMAIL = "morrow@example.invalid";
        GIT_AUTHOR_DATE = "2026-01-01T00:00:00Z";
        GIT_COMMITTER_NAME = "Morrow";
        GIT_COMMITTER_EMAIL = "morrow@example.invalid";
        GIT_COMMITTER_DATE = "2026-01-01T00:00:00Z";
      }
      ''
        export HOME="$TMPDIR"
        git init --quiet --template= --initial-branch=main "$out/repo"
        cd "$out/repo"
        echo morrow > README
        git add README
        git commit --quiet --message start
        git remote add origin https://example.invalid/morrow.git
        git rev-parse HEAD > "$out/root"
      '';
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
    machine.start()
    machine.wait_for_unit("multi-user.target")
    machine.wait_for_unit("user@1000.service")
    systemctl = "runuser -u alice -- env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user"
    machine.wait_until_succeeds(systemctl + " is-active polylogue-main.service")
    pid = int(machine.succeed(systemctl + " show --value --property=MainPID polylogue-main.service"))
    try:
        machine.succeed("runuser -u alice -- ${pkgs.python3}/bin/python3 ${./services.py} ${pkgs.lib.escapeShellArg "${home}/.nix-profile/bin"} " + str(pid) + " ${expectedFile} ${gitFixture}")
    finally:
        machine.succeed(systemctl + " stop polylogue-main.service")
    machine.fail("test -e /proc/" + str(pid))
    machine.succeed("test $(" + systemctl + " show --value --property=ExecMainStatus polylogue-main.service) = 0")
  '';
}
