"""Independent startup-file oracles for the shared deployment renderer."""
import pathlib
import sys
import tomllib


def files(root):
    return {
        path.stem: tomllib.loads(path.read_text())
        for path in (pathlib.Path(root) / "polylogue").glob("*.toml")
    }


assert files(sys.argv[1]) == {
    "polylogue": {
        "archive": "/srv/archive",
        "http": {
            "listen": "127.0.0.1:7788",
            "auth_token_file": "/run/credentials/http",
        },
        "day": {"zone": "Europe/Warsaw", "start": "03:00:00"},
    },
    "signals": {
        "notifications": {
            "away_command": {"program": "/bin/notify", "arguments": ["two words", "--"]},
            "deadline_ms": 1200,
        },
        "value_targets": {"result": {"path": "/var/lib/results/value"}},
        "action_commands": {"open": {"program": "/bin/open", "arguments": ["a b"]}},
    },
}
assert files(sys.argv[2]) == {
    "polylogue": {"archive": "/srv/primary", "http": {}},
    "signals": {"notifications": {}, "value_targets": {}, "action_commands": {}},
    "sinex": {
        "mode": "primary",
        "nats": "nats://127.0.0.1:4222",
        "password_file": "/run/credentials/nats",
        "store": "/var/lib/sinex",
    },
}
