{ pkgs }:
let
  inherit (pkgs) lib;
  settings = import ../lib/settings.nix { inherit lib pkgs; };
  evaluate =
    value:
    (lib.evalModules {
      modules = [
        {
          options.archive = lib.mkOption { type = settings.archiveType; };
          config.archive = value;
        }
      ];
    }).config.archive;
  full = evaluate {
    path = "/srv/archive";
    settings = {
      http = {
        listen = "127.0.0.1:7788";
        auth_token_file = "/run/credentials/http";
      };
      day = {
        zone = "Europe/Warsaw";
        start = "03:00:00";
      };
    };
    signals = {
      notifications = {
        away_command = {
          program = "/bin/notify";
          arguments = [
            "two words"
            "--"
          ];
        };
        deadline_ms = 1200;
      };
      value_targets.result.path = "/var/lib/results/value";
      action_commands.open = {
        program = "/bin/open";
        arguments = [ "a b" ];
      };
    };
  };
  primary = evaluate {
    path = "/srv/primary";
    sinex = {
      mode = "primary";
      nats = "nats://127.0.0.1:4222";
      password_file = "/run/credentials/nats";
      store = "/var/lib/sinex";
    };
  };
  succeeds = value: (builtins.tryEval (builtins.deepSeq value true)).success;
  refused = value: !succeeds (settings.render (evaluate value));
  native = settings.render (evaluate {
    path = "/srv/native";
  });
in
assert lib.assertMsg (
  native == {
    polylogue = {
      archive = "/srv/native";
      http = { };
    };
    signals = {
      notifications = { };
      value_targets = { };
      action_commands = { };
    };
  }
) "Native deployment must leave runtime-owned defaults unspecified.";
assert lib.assertMsg (refused {
  path = "relative/archive";
}) "Relative archive paths must fail evaluation.";
assert lib.assertMsg (refused {
  path = "/archive";
  settings.quiet = true;
}) "Authored settings must not become deployment options.";
assert lib.assertMsg (refused {
  path = "/archive";
  signals.notifications.deadline_ms = 0;
}) "Notification deadlines must be positive.";
assert lib.assertMsg (refused {
  path = "/archive";
  sinex = {
    mode = "primary";
    nats = "nats://localhost";
    store = "/sinex";
  };
  settings.day = {
    zone = "UTC";
    start = "04:00:00";
  };
}) "Primary day ownership must stay with Sinex.";
pkgs.runCommand "polylogue-settings" { nativeBuildInputs = [ pkgs.python3 ]; } ''
  python3 ${./settings.py} ${settings.configDirectory full} ${settings.configDirectory primary}
  touch "$out"
''
