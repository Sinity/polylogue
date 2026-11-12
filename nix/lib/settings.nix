# Both deployment modules use the startup boundary's three file schemas.
{ lib, pkgs }:
let
  inherit (lib) mkOption types;
  absolutePath = types.strMatching "/.*";
  optional =
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
      arguments = optional (types.listOf types.str) "Arguments passed to the executable.";
    };
  };
  deploymentType = types.submodule {
    options = {
      http = mkOption {
        default = { };
        type = types.submodule {
          options = {
            listen = optional types.str "HTTP socket address.";
            auth_token_file = optional absolutePath "File containing the HTTP bearer token.";
          };
        };
      };
      day = optional (types.submodule {
        options = {
          zone = mkOption {
            type = types.str;
            description = "Named time zone for a standalone archive.";
          };
          start = mkOption {
            type = types.str;
            description = "Logical day start as HH:MM:SS.";
          };
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
        description = "Archive storage mode.";
      };
      nats = mkOption {
        type = types.str;
        description = "Credential-free NATS URL.";
      };
      password_file = optional absolutePath "File containing the producer password.";
      subject_root = optional types.str "Sinex subject root.";
      producer = optional types.str "Declared Sinex producer identity.";
      upload_root = optional absolutePath "Local upload scratch when Sinex runs on another host.";
      store = mkOption {
        type = absolutePath;
        description = "Sinex store root on this host.";
      };
    };
  };
  signalsType = types.submodule {
    options = {
      notifications = mkOption {
        default = { };
        type = types.submodule {
          options = {
            away_command = optional commandType "Command that receives each notice sent away, as JSON on standard input.";
            deadline_ms = optional types.ints.positive "Notification delivery deadline in milliseconds.";
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
      sinex = optional sinexType "Sinex deployment. Omit for native standalone storage.";
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
      config,
      data,
      state,
      cache,
      runtime,
    }:
    pkgs.writeShellScriptBin name ''
      export HOME=${lib.escapeShellArg home}
      export XDG_CONFIG_HOME=${lib.escapeShellArg config}
      export XDG_DATA_HOME=${lib.escapeShellArg data}
      export XDG_STATE_HOME=${lib.escapeShellArg state}
      export XDG_CACHE_HOME=${lib.escapeShellArg cache}
      ${
        if runtime == null then
          ''
            export XDG_RUNTIME_DIR="/run/user/$(${pkgs.coreutils}/bin/id -u)"
          ''
        else
          ''
            export XDG_RUNTIME_DIR=${lib.escapeShellArg runtime}
          ''
      }
      exec -a "$0" ${lib.getExe package} --archive ${lib.escapeShellArg archive.path} --config ${configDirectory archive}/polylogue/polylogue.toml "$@"
    '';
}
