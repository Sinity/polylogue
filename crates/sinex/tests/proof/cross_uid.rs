//! Actual CLI publication and provider-free restore through a protected group.
use super::*;
use std::os::unix::fs::PermissionsExt;

#[test]
#[ignore = "needs current POLYLOGUE_BIN/SINEXD/SINEX_SCHEMA, sudo and PostgreSQL/NATS"]
fn cli_publication_and_plain_restore_cross_distinct_uids() {
    let pg = devkit::pg::Server::start(&devkit::pg::ClusterOptions {
        preload: vec!["timescaledb".into()],
        settings: vec![("timescaledb.enable_chunk_skipping".into(), "on".into())],
    })
    .unwrap();
    // Only this owned test cluster's socket parent becomes traversable. Its
    // database files remain private, and no per-user service is selected.
    fs::set_permissions(pg.socket(), fs::Permissions::from_mode(0o711)).unwrap();
    pg.sql("postgres", "CREATE DATABASE sinex").unwrap();
    let user = std::env::var("USER").unwrap();
    let url = pg.url("sinex", Some(&user));
    let schema = std::env::var_os("SINEX_SCHEMA").unwrap();
    assert!(
        Command::new(schema)
            .args(["apply", &url])
            .status()
            .unwrap()
            .success()
    );
    let nats = devkit::nats::Server::start().unwrap();
    let driver = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/proof/cross_uid.py");
    let status = Command::new("sudo")
        .args(["-n", "--preserve-env=PATH", "python3"])
        .arg(driver)
        .arg("--root")
        .arg(std::env::var_os("POLYLOGUE_CROSS_UID_PROOF_DIR").unwrap())
        .arg("--polylogue")
        .arg(std::env::var_os("POLYLOGUE_BIN").unwrap())
        .arg("--sinexd")
        .arg(std::env::var_os("SINEXD").unwrap())
        .args(["--database", &url, "--nats", nats.url()])
        .status()
        .unwrap();
    assert!(
        status.success(),
        "cross-UID command path driver failed: {status}"
    );
}
