# Both deployment modules use the startup boundary's three file schemas.
{ lib, pkgs }:
let
  inherit (lib) mkOption types;
  absolutePath = types.strMatching "/.*";
  nullableOption =
    type: description:
    mkOption {
      type = types.nullOr type;
      default = null;
      inherit description;
    };
  commandType = types.submodule {
    options = {
      program = mkOption {
        type = absolutePath;
        description = "Executable path. Arguments are passed separately.";
      };
      arguments = nullableOption (types.listOf types.str) "Arguments passed to the executable.";
    };
  };
  deploymentType = types.submodule {
    options = {
      http = mkOption {
        default = { };
        type = types.submodule {
          options = {
            listen = nullableOption types.str "HTTP socket address.";
            auth_token_file = nullableOption absolutePath "File containing the HTTP bearer token.";
          };
        };
      };
      day = nullableOption (types.submodule {
        options = {
          zone = nullableOption types.str "Named time zone for a standalone archive. Omit for the host's zone.";
          start = nullableOption types.str "Logical day start as HH:MM:SS. Omit for 04:00:00.";
        };
      }) "Standalone logical day. Primary mode obtains its day from Sinex.";
    };
  };
  sinexType = types.submodule {
    options = {
      mode = mkOption {
        type = types.enum [
          "standalone"
          "primary"
        ];
        description = "Archive storage mode. Primary mode is not built in this version: Polylogue refuses to start with it.";
      };
    };
  };
  signalsType = types.submodule {
    options = {
      notifications = mkOption {
        default = { };
        type = types.submodule {
          options = {
            away_command = nullableOption commandType "Command that receives each notice the route function sends away, as JSON on standard input.";
            deadline_ms = nullableOption types.ints.positive "Notification delivery deadline in milliseconds.";
          };
        };
      };
      value_targets = mkOption {
        default = { };
        type = types.attrsOf (
          types.submodule {
            options.path = mkOption {
              type = absolutePath;
              description = "File the named signal outlet may update.";
            };
          }
        );
        description = "Named value outlets available to authored rules.";
      };
      action_commands = mkOption {
        default = { };
        type = types.attrsOf commandType;
        description = "Named action commands available to authored rules.";
      };
    };
  };
  stripNulls =
    value:
    if builtins.isAttrs value then
      lib.mapAttrs (_: stripNulls) (lib.filterAttrs (_: item: item != null) value)
    else if builtins.isList value then
      map stripNulls value
    else
      value;
  render =
    archive:
    assert lib.assertMsg (
      archive.sinex == null || archive.sinex.mode != "primary" || archive.settings.day == null
    ) "Primary Polylogue archives obtain their logical day from Sinex; omit settings.day.";
    stripNulls (
      {
        polylogue = archive.settings // {
          archive = archive.path;
        };
        signals = archive.signals;
      }
      // lib.optionalAttrs (archive.sinex != null) { sinex = archive.sinex; }
    );
  configDirectory =
    archive:
    pkgs.linkFarm "polylogue-config" (
      lib.mapAttrsToList (name: value: {
        name = "polylogue/${name}.toml";
        path = (pkgs.formats.toml { }).generate "${name}.toml" value;
      }) (render archive)
    );
in
{
  inherit render configDirectory;
  archiveType = types.submodule {
    options = {
      path = mkOption {
        type = absolutePath;
        description = "Archive directory, passed explicitly to every invocation.";
      };
      settings = mkOption {
        type = deploymentType;
        default = { };
        description = "Deployment values accepted by polylogue.toml.";
      };
      sinex = nullableOption sinexType "Sinex deployment. Omit for standalone storage.";
      signals = mkOption {
        type = signalsType;
        default = { };
        description = "Signal delivery outlets. Attention preferences remain authored settings.";
      };
    };
  };
  # The installed name is stable across configuration changes. Services use
  # that profile path and put only the application in their restart trigger.
  wrapper =
    {
      name,
      package,
      archive,
      home,
      configHome,
      dataHome,
      stateHome,
      cacheHome,
      runtimeDir,
    }:
    pkgs.writeShellScriptBin name ''
      export HOME=${lib.escapeShellArg home}
      export XDG_CONFIG_HOME=${lib.escapeShellArg configHome}
      export XDG_DATA_HOME=${lib.escapeShellArg dataHome}
      export XDG_STATE_HOME=${lib.escapeShellArg stateHome}
      export XDG_CACHE_HOME=${lib.escapeShellArg cacheHome}
      ${
        if runtimeDir == null then
          ''
            export XDG_RUNTIME_DIR="/run/user/$(${pkgs.coreutils}/bin/id -u)"
          ''
        else
          ''
            export XDG_RUNTIME_DIR=${lib.escapeShellArg runtimeDir}
          ''
      }
      exec -a "$0" ${lib.getExe package} --archive ${lib.escapeShellArg archive.path} --config ${configDirectory archive}/polylogue/polylogue.toml "$@"
    '';
}
