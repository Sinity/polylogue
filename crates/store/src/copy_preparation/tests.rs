use super::*;
use crate::{ReadHandle, Writer};
const A: &str = "019a1100-0000-7000-8000-000000000001";
const B: &str = "019a1100-0000-7000-8000-000000000002";
#[test]
fn interrupted_preparation_fences_writer_and_resumes_same_action_without_repeating_completion() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    writer.put_file_blob(b"").unwrap();
    writer.put_file_blob(b"retained content").unwrap();
    let before = writer.status().unwrap();
    drop(writer);
    let stop = AtomicBool::new(false);
    let prep = Preparation::begin(root.path(), "restore", "local", A, true, &stop).unwrap();
    assert_eq!(prep.progress()["outcome"], "pending");
    assert!(Writer::open(root.path()).is_err());
    stop.store(true, std::sync::atomic::Ordering::Release);
    assert!(matches!(prep.finish(&stop), Err(Error::Cancelled)));
    assert!(
        Writer::open(root.path())
            .unwrap_err_for_test()
            .contains("pending action")
    );
    stop.store(false, std::sync::atomic::Ordering::Release);
    assert!(Preparation::begin(root.path(), "restore", "local", B, true, &stop).is_err());
    let report = Preparation::begin(root.path(), "restore", "local", A, true, &stop)
        .unwrap()
        .finish(&stop)
        .unwrap();
    assert_eq!(report.action, "prepared");
    let writer = Writer::open(root.path()).unwrap();
    let after = writer.status().unwrap();
    assert_eq!(before.acquisition_lineage, after.acquisition_lineage);
    assert_eq!(after.write_counter, before.write_counter + 2);
    assert_eq!(after.unavailable_blob_files, 0);
    drop(writer);
    assert_eq!(
        Preparation::begin(root.path(), "restore", "local", B, true, &stop)
            .unwrap()
            .finish(&stop)
            .unwrap()
            .action,
        "prepared"
    );
    let writer = Writer::open(root.path()).unwrap();
    let counter = writer.status().unwrap().write_counter;
    drop(writer);
    assert_eq!(
        Preparation::begin(root.path(), "restore", "local", A, true, &stop)
            .unwrap()
            .finish(&stop)
            .unwrap()
            .action,
        "historical_completion"
    );
    assert_eq!(
        Writer::open(root.path())
            .unwrap()
            .status()
            .unwrap()
            .write_counter,
        counter
    );
}
#[test]
fn unsupported_modes_missing_custody_and_missing_historical_lineage_cannot_mint_authority() {
    let root = tempfile::tempdir().unwrap();
    let writer = Writer::open(root.path()).unwrap();
    drop(writer);
    let stop = AtomicBool::new(false);
    for (mode, backing, retired) in [
        ("independent", "local", true),
        ("restore", "primary", true),
        ("restore", "local", false),
    ] {
        assert!(Preparation::begin(root.path(), mode, backing, A, retired, &stop).is_err());
    }
    let path = root.path().join("archive.db");
    let conn = open(&path, false, true).unwrap();
    conn.execute("DELETE FROM meta WHERE key='acquisition_lineage'", [])
        .unwrap();
    drop(conn);
    assert!(Preparation::begin(root.path(), "restore", "local", A, true, &stop).is_err());
    assert!(Writer::open(root.path()).is_err());
    let reader = ReadHandle::open(root.path()).unwrap();
    assert!(reader.status().unwrap().acquisition_lineage.is_none());
}
#[test]
fn missing_blob_blocks_completion_and_status_keeps_unrelated_readable_bytes() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(root.path()).unwrap();
    let missing = writer.put_file_blob(b"missing").unwrap();
    let good = writer.put_file_blob(b"unrelated").unwrap();
    drop(writer);
    let stop = AtomicBool::new(false);
    Preparation::begin(root.path(), "restore", "local", A, true, &stop)
        .unwrap()
        .finish(&stop)
        .unwrap();
    let name = missing.to_string();
    fs::remove_file(
        root.path()
            .join("blobs")
            .join(&name[..2])
            .join(format!("{name}.zst")),
    )
    .unwrap();
    let writer = Writer::open(root.path()).unwrap();
    let reader = writer.read_handle().unwrap();
    assert_eq!(reader.status().unwrap().unavailable_blob_files, 1);
    assert_eq!(reader.read_blob_range(good, 0, 9).unwrap(), b"unrelated");
    assert!(reader.read_blob_range(missing, 0, 7).is_err());
    drop(reader);
    drop(writer);
    let prep = Preparation::begin(root.path(), "restore", "local", B, true, &stop).unwrap();
    assert!(prep.finish(&stop).is_err());
    assert!(Writer::open(root.path()).is_err());
}
trait Refused {
    fn unwrap_err_for_test(self) -> String;
}
impl Refused for Result<Writer> {
    fn unwrap_err_for_test(self) -> String {
        match self {
            Ok(_) => panic!("unexpected writer"),
            Err(e) => e.to_string(),
        }
    }
}
