{
  pkgs,
  polylogue,
  revision,
}:
pkgs.testers.runNixOSTest {
  name = "polylogue-package-behavior";
  nodes.machine = {
    # The daemon takes the logical day's zone from the host's named zone.
    time.timeZone = "UTC";
    environment.systemPackages = [
      polylogue
      pkgs.python3
    ];
  };
  testScript = ''
    machine.start()
    machine.wait_for_unit("multi-user.target")
    machine.succeed("${pkgs.python3}/bin/python3 ${./behavior.py} ${polylogue}/bin/polylogue ${pkgs.lib.escapeShellArg revision}")
  '';
}
