//! The real primary-mode fixture. Every database, bus, credential and store is synthetic.
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};

struct Daemon(Child);
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub struct Native {
    pub socket: PathBuf,
    pub store: PathBuf,
    pub nats: devkit::nats::Server,
    pub pg: devkit::pg::Server,
    root: PathBuf,
    daemon: Option<Daemon>,
}
impl Native {
    pub fn start(root: &Path, declarations: Vec<serde_json::Value>) -> Self {
        fs::create_dir_all(root).unwrap();
        let pg = devkit::pg::Server::start(&devkit::pg::ClusterOptions {
            required_extensions: vec!["timescaledb".into(), "vector".into()],
            preload: vec!["timescaledb".into()],
            settings: vec![("timescaledb.enable_chunk_skipping".into(), "on".into())],
        })
        .unwrap();
        pg.sql("postgres", "CREATE DATABASE sinex").unwrap();
        let status = Command::new(binary())
            .args(["admin", "schema", "--database", &pg.url("sinex", None)])
            .status()
            .unwrap();
        assert!(status.success(), "sinexd admin schema");
        fs::create_dir_all(root.join("tokens")).unwrap();
        for role in ["read", "agent", "operator"] {
            fs::write(
                root.join("tokens").join(format!("{role}.token")),
                format!("{role}-rebuild-token"),
            )
            .unwrap();
        }
        use sinex_model::EventPayload;
        let evidence_source = sinex_model::SourceDeclaration {
            source: sinex_model::SourceRef {
                name: "synthetic-evidence".into(),
                version: 1,
            },
            family: "synthetic-evidence".into(),
            kinds: vec![
                sinex_model::event_types::session::TerminalReport::declaration(),
                sinex_model::event_types::machine::FileChanged::declaration(),
                sinex_model::event_types::work::WorkAct::declaration(),
            ],
        };
        let grants = serde_json::json!({"polylogue":{
            "kind_prefixes":["polylogue"],"classes":["open","guarded"],"public_context":true,
            "write":{"family":"polylogue","host":"fixture-host","source_prefixes":["polylogue"],
                "source_declarations":declarations,"kind_prefixes":["polylogue"],
                "canonical_prefixes":["polylogue"],"classes":["open","guarded"]}
        }, "evidence": {
            "kind_prefixes":["terminal","file","work"],"classes":["open"],"public_context":true,
            "write":{"family":"synthetic-evidence","host":"fixture-host",
                "source_prefixes":["synthetic-evidence"],"source_declarations":[evidence_source],
                "kind_prefixes":["terminal","file","work"],
                "canonical_prefixes":["terminal","file","work"],"classes":["open"]}
        }});
        fs::write(
            root.join("grants.json"),
            serde_json::to_vec(&grants).unwrap(),
        )
        .unwrap();
        let mut native = Self {
            socket: root.join("sinexd.sock"),
            store: root.join("sinex-store"),
            nats: devkit::nats::Server::start().unwrap(),
            pg,
            root: root.into(),
            daemon: None,
        };
        native.restart();
        native
    }

    pub fn restart(&mut self) {
        assert!(self.daemon.is_none());
        self.daemon = Some(Daemon(
            polylogue_testkit::spawn_daemon(
                Command::new(binary())
                    .arg("run")
                    .args(["--database", &self.pg.url("sinex", None)])
                    .arg("--store")
                    .arg(&self.store)
                    .args([
                        "--nats",
                        self.nats.url(),
                        "--subject-root",
                        "rebuild",
                        "--zone",
                        "UTC",
                    ])
                    .arg("--producer-grants")
                    .arg(self.root.join("grants.json"))
                    .arg("--tokens")
                    .arg(self.root.join("tokens"))
                    .arg("--socket")
                    .arg(&self.socket),
            )
            .unwrap(),
        ));
    }

    pub fn stop(&mut self) {
        drop(self.daemon.take());
    }

    pub fn sql(&self, sql: &str) -> String {
        self.pg.sql("sinex", sql).unwrap()
    }

    /// Restore both actual native dumps into an empty database. A new bus
    /// prevents post-cut JetStream deliveries from masquerading as restored data.
    pub fn restore(&mut self, artifact: &Path, restored_store: &Path) {
        self.stop();
        self.pg
            .sql("postgres", "DROP DATABASE sinex WITH (FORCE)")
            .unwrap();
        self.pg
            .sql("postgres", "CREATE DATABASE sinex TEMPLATE template0")
            .unwrap();
        let status = Command::new(binary())
            .args([
                "admin",
                "prepare-database",
                "--database",
                &self.pg.url("sinex", None),
                "--owner",
                &std::env::var("USER").unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success(), "prepare empty restore database");
        self.sql("SELECT timescaledb_pre_restore()");
        for (name, data_only) in [("main/sinex.dump", false), ("keys/data-keys.dump", true)] {
            let mut command = Command::new("pg_restore");
            command.args([
                "--dbname",
                &self.pg.url("sinex", None),
                "--no-password",
                "--no-owner",
                "--exit-on-error",
            ]);
            if data_only {
                command.arg("--data-only");
            }
            let status = command.arg(artifact.join(name)).status().unwrap();
            assert!(status.success(), "restore {name}");
        }
        self.sql("SELECT timescaledb_post_restore()");
        fs::create_dir_all(restored_store).unwrap();
        let cancelled = std::sync::atomic::AtomicBool::new(false);
        sinex_transport::backup::capture_tree(
            &artifact.join("main/segments"),
            &restored_store.join("segments"),
            &cancelled,
        )
        .unwrap();
        sinex_transport::backup::capture_tree(
            &artifact.join("main/producers/polylogue"),
            restored_store,
            &cancelled,
        )
        .unwrap();
        self.store = restored_store.into();
        self.nats = devkit::nats::Server::start().unwrap();
    }
}

fn binary() -> String {
    std::env::var("SINEXD").expect("SINEXD names the matching built public Sinex daemon")
}

// Native backup also latches the daemon telemetry generation. Its finite
// natural seal may take 300 seconds; use the native fixture's 600-second budget.
pub fn request(socket: &Path, role: &str, body: serde_json::Value) -> serde_json::Value {
    let body = body.to_string();
    let mut stream = std::os::unix::net::UnixStream::connect(socket).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(600)))
        .unwrap();
    write!(stream,"POST /v1 HTTP/1.1\r\nHost: sinexd\r\nContent-Type: application/json\r\nAuthorization: Bearer {role}-rebuild-token\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).unwrap();
    let split = reply
        .windows(4)
        .position(|part| part == b"\r\n\r\n")
        .expect("HTTP headers");
    let headers = String::from_utf8_lossy(&reply[..split]).to_ascii_lowercase();
    let mut body = &reply[split + 4..];
    let mut decoded = Vec::new();
    if headers.contains("transfer-encoding: chunked") {
        loop {
            let end = body.windows(2).position(|part| part == b"\r\n").unwrap();
            let size = usize::from_str_radix(
                std::str::from_utf8(&body[..end])
                    .unwrap()
                    .split(';')
                    .next()
                    .unwrap(),
                16,
            )
            .unwrap();
            body = &body[end + 2..];
            if size == 0 {
                break;
            }
            decoded.extend_from_slice(&body[..size]);
            body = &body[size + 2..];
        }
        body = &decoded;
    }
    serde_json::from_slice(body)
        .unwrap_or_else(|error| panic!("{error}: {}", String::from_utf8_lossy(&reply)))
}
