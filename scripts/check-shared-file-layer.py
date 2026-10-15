"""Qualify an extracted package and a disposable local-tag consumer."""

import json
import os
import re
from pathlib import Path
import subprocess
import tarfile
import tempfile
import tomllib
import argparse


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", choices=["hashed-blobs", "credential-output"], default="hashed-blobs")
NAME = parser.parse_args().package
PACKAGE = ROOT / "crates" / NAME
MANIFEST = tomllib.loads((PACKAGE / "Cargo.toml").read_text())["package"]
TAG = f"{NAME}-v{MANIFEST['version']}"


def run(*args, cwd=ROOT, capture=False, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(
        list(map(str, args)), cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE if capture else None, env=env,
    ).stdout


def check_dependencies(manifest, cwd):
    metadata = json.loads(run(
        "cargo", "metadata", "--locked", "--offline", "--format-version", "1",
        "--manifest-path", manifest, cwd=cwd, capture=True,
    ))
    names = {p["name"] for p in metadata["packages"]}
    forbidden = {
        name for name in names
        if name in {"tokio", "rusqlite", "libsqlite3-sys", "async-nats", "notify", "pest"}
        or name.startswith(("sqlx", "sinex-", "polylogue-"))
    }
    assert not forbidden, sorted(forbidden)
    print("standalone dependency graph:", ", ".join(sorted(names)), flush=True)


assert not run("git", "status", "--porcelain", capture=True).strip(), "commit the candidate first"
revision = run("git", "rev-parse", "HEAD", capture=True).strip()
compiler = run("rustc", "--version", capture=True).strip()
actual = re.match(r"^rustc ([0-9]+)\.([0-9]+)\.([0-9]+)(?:\s|$)", compiler)
minimum = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)", MANIFEST["rust-version"])
assert actual and minimum, (compiler, MANIFEST["rust-version"])
assert tuple(map(int, actual.groups())) == tuple(map(int, minimum.groups())), (
    "minimum-tested qualification must run the exact declared stable compiler", compiler, MANIFEST["rust-version"]
)
print("candidate:", revision, "compiler:", compiler, "declared minimum:", MANIFEST["rust-version"], flush=True)

with tempfile.TemporaryDirectory(prefix="shared-file-layer-", dir=os.environ["TMPDIR"]) as directory:
    scratch = Path(directory)
    target = scratch / "target"
    env = os.environ | {"CARGO_TARGET_DIR": str(target)}
    run("cargo", "package", "--locked", "--offline", "--no-verify", "-p", NAME, env=env)
    archive = target / "package" / f"{NAME}-{MANIFEST['version']}.crate"
    with tarfile.open(archive) as package:
        package.extractall(scratch, filter="data")
    extracted = scratch / f"{NAME}-{MANIFEST['version']}"
    assert (extracted / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert (extracted / "src/lib.rs").read_bytes() == (PACKAGE / "src/lib.rs").read_bytes()
    check_dependencies(extracted / "Cargo.toml", extracted)
    run("cargo", "test", "--locked", "--offline", cwd=extracted, env=env)

    # Tags exist only inside this disposable clone. No tag/ref in the owner changes.
    owner = scratch / "owner"
    run("git", "clone", "--shared", "--no-checkout", ROOT, owner)
    run("git", "-C", owner, "-c", "tag.gpgSign=false", "tag", TAG, revision)
    consumer = scratch / "consumer"
    (consumer / "src").mkdir(parents=True)
    (consumer / "Cargo.toml").write_text(
        '[package]\nname = "shared-file-consumer"\nversion = "0.0.0"\nedition = "2021"\n'
        '[dependencies]\n'
        f'{NAME} = {{ git = "{owner.as_uri()}", tag = "{TAG}" }}\n'
    )
    (consumer / "src/main.rs").write_text('''use hashed_blobs::{DataKey, Hash, Reader, Store};
use std::io::{Cursor, Read};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args_os().nth(1).expect("synthetic root");
    let bytes = b"synthetic cross-consumer bytes";
    let writer = Store::open(root.clone(), vec![], 0o600)?.with_frame_size(8);
    let name = writer.put(Cursor::new(bytes), 1)?;
    assert_eq!(name, Hash::of(bytes));
    let reader = Reader::new(root, vec![]);
    assert_eq!(reader.read_range(&name, 10, 5)?, &bytes[10..15]);
    let mut all = Vec::new();
    reader.read(&name)?.read_to_end(&mut all)?;
    assert_eq!(all, bytes);
    let key = DataKey { key_ref: "synthetic-key".into(), key: [7; 32] };
    let sealed = writer.put_sealed(Cursor::new(bytes), 1, &key)?;
    assert!(writer.read(&sealed).is_err());
    assert_eq!(writer.read_range_with(&sealed, 10, 5, &|_| Ok([7; 32]))?, &bytes[10..15]);
    Ok(())
}
''')
    if NAME == "credential-output":
        (consumer / "src/main.rs").write_text('''use credential_output::Masker;
fn main() -> std::io::Result<()> {
    let mut masker = Masker::new();
    let mut out = Vec::new();
    masker.push("before ghp_", &mut out)?;
    masker.push(&"A".repeat(36), &mut out)?;
    masker.push(" after", &mut out)?;
    masker.finish(&mut out)?;
    assert_eq!(String::from_utf8(out).unwrap(), "before [credential: GitHub token, masked] after");
    Ok(())
}
''')
    # First resolution fetches the local git URL. Registry dependencies are cached by packaging.
    metadata = json.loads(run("cargo", "metadata", "--format-version", "1", cwd=consumer, capture=True))
    blob = next(p for p in metadata["packages"] if p["name"] == NAME)
    assert blob["source"].endswith("#" + revision), blob["source"]
    check_dependencies(consumer / "Cargo.toml", consumer)
    run("cargo", "run", "--locked", "--offline", "--", scratch / "plain", cwd=consumer, env=env)
    run("cargo", "run", "--locked", "--offline", "--no-default-features", "--", scratch / "no-default", cwd=consumer, env=env)
    print("disposable local tag resolved exact candidate:", revision, flush=True)
print(NAME + ": extracted tests, license, dependency isolation and local-tag consumer passed", flush=True)
