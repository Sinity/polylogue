"""One synthetic CLI publisher/receiver/restore control; invoked by proof.rs."""
import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import signal
import sqlite3
import subprocess
import time
import zipfile

SENDER, RECEIVER, READER, GROUP = 60001, 60002, 60004, 60010
SESSION = "chatgpt:group-command-session"
MESSAGE = "chatgpt:group-command-message"
NOTE = "Synthetic exact group note λ🙂\nsecond line"
ASSET = bytes(range(256)) * 1024
SET = "/run/current-system/sw/bin/setpriv"


def main():
    p = argparse.ArgumentParser()
    for name in ("root", "polylogue", "sinexd", "database", "nats"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    assert os.geteuid() == 0
    root = pathlib.Path(a.root)
    root.mkdir(mode=0o711)
    os.chmod(root, 0o711)
    # SQLite resolves proc-fd paths to their inaccessible physical parents.
    # A private mount gives every actor a normal path without changing those
    # parent permissions or adding executable copies.
    namespace_before = os.readlink("/proc/self/ns/mnt")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(0x00020000) != 0:
        raise OSError(ctypes.get_errno(), "private mount namespace")
    subprocess.run(["mount", "--make-rprivate", "/"], check=True)
    for name, binary in (("polylogue", a.polylogue), ("sinexd", a.sinexd)):
        os.link(binary, root / name)
    subprocess.run(["mount", "--bind", str(root), "/mnt"], check=True)
    visible = pathlib.Path("/mnt")
    namespace_after = os.readlink("/proc/self/ns/mnt")
    assert namespace_before != namespace_after
    def digest(path):
        with open(path, "rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    pins = json.loads((root.parent / "binary-pins.json").read_text())
    binary_hashes = {}
    for name, binary in (("polylogue", a.polylogue), ("sinexd", a.sinexd)):
        binary_hashes[name] = digest(binary)
        assert binary_hashes[name] == pins[name]["sha256"]
    result = None
    processes = []
    transcript = []
    terminations = []

    def directory(name, uid, gid, mode):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, uid, gid)
        os.chmod(path, mode)
        return path

    def actor(uid, gid, args):
        groups = f"--groups={GROUP}" if uid != RECEIVER else "--clear-groups"
        return [SET, "--clear-groups", "--", SET, "--reuid", str(uid), "--regid", str(gid), groups, "--", *map(str, args)]

    def env(uid):
        result = os.environ.copy()
        result.update(POLYLOGUE_HOME=str(visible / f"home-{uid}"),
                      POLYLOGUE_ARCHIVE=str(visible / f"archive-{uid}"),
                      XDG_CONFIG_HOME=str(visible / f"cfg-{uid}"),
                      XDG_RUNTIME_DIR=str(visible / f"run-{uid}"),
                      TMPDIR=str(visible / f"tmp-{uid}"))
        return result

    def run(uid, args, binary=a.polylogue, timeout=15, check=True):
        gid = GROUP if uid == RECEIVER else uid
        binary = str(visible / "polylogue") if binary == a.polylogue else str(visible / "sinexd") if binary == a.sinexd else binary
        out = subprocess.run(actor(uid, gid, [binary, *args]), env=env(uid), cwd=visible, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        transcript.append({"uid": uid, "args": list(map(str, args)), "code": out.returncode,
                           "stdout": out.stdout.decode(errors="replace"),
                           "stderr": out.stderr.decode(errors="replace")})
        if check:
            assert out.returncode == 0, transcript[-1]
        return out

    def cli(uid, args):
        return json.loads(run(uid, ["--json", *args]).stdout)

    def start(uid, args, binary=a.polylogue):
        log = open(root / f"process-{uid}-{len(processes)}.log", "wb")
        gid = GROUP if uid == RECEIVER else uid
        binary = str(visible / "polylogue") if binary == a.polylogue else str(visible / "sinexd") if binary == a.sinexd else binary
        proc = subprocess.Popen(actor(uid, gid, [binary, *args]), env=env(uid), cwd=visible, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, start_new_session=True)
        processes.append((proc, log))
        return proc

    def stop(proc):
        if proc.poll() is not None:
            return
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f"owned actor group still exists: {proc.pid}")
        terminations.append({"pid": proc.pid, "policy": "owned crash/reap",
                             "returncode": proc.returncode, "group_reaped": True})
        (root / "termination.json").write_text(json.dumps(terminations, indent=2) + "\n")

    def wait(predicate, label, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.1)
        raise AssertionError("timed out: " + label)

    def ready(uid, proc):
        def probe():
            assert proc.poll() is None, f"actor {uid} exited"
            if not (root / f"archive-{uid}/daemon.sock").exists():
                return False
            return run(uid, ["--json", "status"], check=False).returncode == 0
        wait(probe, f"Polylogue {uid} startup")

    def sql(statement):
        out = subprocess.run(["psql", a.database, "-At", "-c", statement],
                             capture_output=True, check=True, timeout=10)
        return out.stdout.decode().strip()

    def db(path, statement):
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            return connection.execute(statement).fetchall()

    shared = directory("shared", RECEIVER, GROUP, 0o2770)
    for name in ("incoming", "incoming/producers", "incoming/producer-requests"):
        directory("shared/" + name, RECEIVER, GROUP, 0o2770)
    for uid in (SENDER, RECEIVER, READER):
        for prefix in ("home", "cfg", "run", "tmp", "archive"):
            directory(f"{prefix}-{uid}", uid, GROUP if uid == RECEIVER else uid, 0o700)
    inputs = directory("inputs", SENDER, SENDER, 0o700)
    path = inputs / "original.zip"
    conversation = [{"id": "group-command-session", "title": "Synthetic group ZIP", "mapping": {
        "n": {"id": "n", "parent": None, "children": [], "message": {
            "id": "group-command-message", "author": {"role": "user"},
            "content": {"content_type": "multimodal_text", "parts": [
                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-Group"}]},
            "metadata": {}}}}, "current_node": "n"}]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as output:
        output.comment = "Synthetic original ZIP metadata λ🙂".encode()
        output.writestr("conversations.json", json.dumps(conversation).encode())
        output.writestr("file-Group.dat", ASSET)
        output.writestr("conversation_asset_file_names.json", b'{"file-Group.dat":"group-original.dat"}')
    os.chown(path, SENDER, SENDER)
    original = path.read_bytes()
    runtime = root / f"run-{RECEIVER}"
    tokens = runtime / "tokens"
    tokens.mkdir()
    os.chown(tokens, RECEIVER, GROUP)
    os.chmod(tokens, 0o700)
    for name in ("read", "agent", "operator"):
        token = tokens / (name + ".token")
        token.write_text("synthetic-" + name)
        os.chown(token, RECEIVER, GROUP)
        os.chmod(token, 0o600)

    def configure(uid, archive=None):
        cfg = root / f"cfg-{uid}/polylogue"
        cfg.mkdir(exist_ok=True)
        os.chown(cfg, uid, uid)
        data = dict(mode="primary", nats=a.nats, subjects="crossuid", producer="polylogue", store=str(visible / "shared"))
        if archive:
            data["archive"] = archive
        target = cfg / "sinex.toml"
        target.write_text("\n".join(k + "=" + json.dumps(v) for k, v in data.items()) + "\n")
        os.chown(target, uid, uid)
        os.chmod(target, 0o600)

    try:
        for uid in (SENDER, RECEIVER, READER):
            assert run(uid, ["-u"], binary="id").stdout.decode().strip() == str(uid)
            groups = sorted(set(map(int, run(uid, ["-G"], binary="id").stdout.split())))
            assert groups == ([GROUP] if uid == RECEIVER else sorted([uid, GROUP]))
        # The source archive is created only by the ordinary CLI and Writer.
        sender = start(SENDER, ["daemon"])
        ready(SENDER, sender)
        imported = cli(SENDER, ["import", str(visible / "inputs/original.zip")])
        assert not imported["pass"]["errors"], imported
        cli(SENDER, ["note", "add", "message:" + MESSAGE, "--body", NOTE])
        expected_read = cli(SENDER, ["read", SESSION])
        expected_notes = cli(SENDER, ["note", "list", "message:" + MESSAGE])
        stop(sender)
        source = root / f"archive-{SENDER}/archive.db"
        archive = db(source, "SELECT value FROM meta WHERE key='archive_id'")[0][0]
        containers = db(source, "SELECT DISTINCT container_input FROM observation_container")
        assert len(containers) == 1, containers
        original_id = containers[0][0]
        expected_note_rows = db(source, "SELECT * FROM note ORDER BY id")
        assert db(source, "SELECT value FROM note") == [(NOTE,)]
        counts = {table: db(source, "SELECT COUNT(*) FROM " + table)[0][0] for table in
                  ("revision", "observation", "note", "setting", "delivery", "delivery_event", "excision", "extracted_blob")}
        expected_units = sum(counts.values())
        assert expected_units > 3 and counts["note"] == 1, counts
        receiver_args = ["run", "--database", a.database, "--store", str(visible / "shared"),
                         "--nats", a.nats, "--subjects", "crossuid", "--tokens", str(visible / f"run-{RECEIVER}/tokens"),
                         "--socket", str(visible / f"run-{RECEIVER}/sinexd.sock")]
        receiver = start(RECEIVER, receiver_args, binary=a.sinexd)
        wait(lambda: (runtime / "sinexd.sock").exists(), "Sinex receiver startup")
        configure(SENDER)
        sender = start(SENDER, ["daemon"])
        ready(SENDER, sender)
        prefix = "rows:" + archive + ":%"
        wait(lambda: int(sql("SELECT count(*) FROM core.ingress_frames WHERE producer='polylogue' AND refusal IS NULL AND frame LIKE '" + prefix + "'")) == expected_units,
             "actual CLI publisher final acknowledgments", 45)
        stop(sender)
        frames_before = sql("SELECT count(*) FROM core.ingress_frames WHERE producer='polylogue'")
        # Distinct receiver owns the public file; a fresh distinct group client
        # must restore it without source-archive or provider access.
        retained = directory("retained-provider", 0, 0, 0o700)
        os.rename(path, retained / "original.zip")
        configure(READER, archive)
        reader = start(READER, ["daemon"])
        ready(READER, reader)
        actual_read = cli(READER, ["read", SESSION])
        actual_notes = cli(READER, ["note", "list", "message:" + MESSAGE])
        assert actual_read == expected_read, (actual_read, expected_read)
        assert actual_notes == expected_notes, (actual_notes, expected_notes)
        restored = root / f"archive-{READER}/archive.db"
        assert db(restored, "SELECT DISTINCT container_input FROM observation_container") == [(original_id,)]
        assert db(restored, "SELECT * FROM note ORDER BY id") == expected_note_rows
        blob = shared / "blobs" / original_id[:2] / (original_id + ".zst")
        stat = blob.stat()
        assert (stat.st_uid, stat.st_gid, stat.st_mode & 0o777) == (RECEIVER, GROUP, 0o640), (stat.st_uid, stat.st_gid, oct(stat.st_mode))
        decoded = run(READER, ["-d", "-q", "-c", str(visible / blob.relative_to(root))], binary="zstd").stdout
        assert decoded == original
        denied = run(READER, ["-r", str(visible / source.relative_to(root))], binary="test", check=False)
        assert denied.returncode != 0, "restore actor could read source archive"
        stop(reader)
        stop(receiver)
        (runtime / "sinexd.sock").unlink(missing_ok=True)
        os.rename(root / f"archive-{READER}", root / "retained-first-restore")
        directory(f"archive-{READER}", READER, READER, 0o700)
        receiver = start(RECEIVER, receiver_args, binary=a.sinexd)
        wait(lambda: (runtime / "sinexd.sock").exists() and receiver.poll() is None, "receiver restart")
        reader = start(READER, ["daemon"])
        ready(READER, reader)
        assert cli(READER, ["read", SESSION]) == expected_read
        assert cli(READER, ["note", "list", "message:" + MESSAGE]) == expected_notes
        assert sql("SELECT count(*) FROM core.ingress_frames WHERE producer='polylogue'") == frames_before
        result = {"result": "passed", "uids": {"publisher": SENDER, "receiver": RECEIVER, "restorer": READER},
                  "binary_sha256": binary_hashes, "group": GROUP, "mount_namespace": {"before": namespace_before, "isolated": namespace_after}, "archive": archive, "units": expected_units, "original_zip": original_id,
                  "original_zip_sha256": hashlib.sha256(original).hexdigest(), "note": NOTE,
                  "provider_path_absent": not path.exists(), "source_archive_denied_to_reader": True,
                  "receiver_file_uid": stat.st_uid, "receiver_file_mode": oct(stat.st_mode & 0o777),
                  "scope": "synthetic current CLI publisher/receiver/provider-free restore/restart; no installed operation or cutover claim"}
    finally:
        for proc, log in reversed(processes):
            stop(proc)
            log.close()
        # Byte output is synthetic, but keep command transcripts bounded.
        for item in transcript:
            if len(item["stdout"]) > 10000:
                item["stdout"] = "[binary output retained/compared separately]"
        (root / "commands.json").write_text(json.dumps(transcript, indent=2) + "\n")
    for name, binary in (("polylogue", a.polylogue), ("sinexd", a.sinexd)):
        assert digest(binary) == binary_hashes[name]
    assert result is not None
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
