//! The configured main may acquire explicitly referenced tool-result files in
//! its own provider directory. No directory discovery or attachment traversal.
use polylogue_store::{Error, ImportObservation, StoredSource, Writer};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
pub fn acquire(
    writer: &mut Writer,
    source: &StoredSource,
    main: &Path,
    machine: &str,
    host: &str,
    request: &str,
    seen_at: i64,
    cancelled: &AtomicBool,
) -> Result<StoredSource, Error> {
    if cancelled.load(Ordering::Acquire) {
        return Err(Error::Cancelled);
    }
    let reader = writer.read_handle()?;
    let mut pending = Vec::new();
    let mut paths = BTreeMap::new();
    let mut offset = 0;
    while offset < source.input.length {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let bytes = reader.read_source_range(
            &source.input.id,
            source.input.chain,
            source.binding,
            offset,
            (source.input.length - offset).min(65536),
        )?;
        offset += bytes.len() as u64;
        for byte in bytes {
            if byte == b'\n' {
                if let Ok(record) = polylogue_model::RawJson::parse(&pending) {
                    for (path, hash, length) in
                        polylogue_sources::claude_code::body_references(&record)
                    {
                        paths
                            .entry(path)
                            .and_modify(|old| {
                                if *old != Some((hash, length)) {
                                    *old = None;
                                }
                            })
                            .or_insert(Some((hash, length)));
                    }
                }
                pending.clear();
            } else {
                if pending.len() >= 1024 * 1024 {
                    return Err(Error::ObservedSourceRefused(
                        "Claude companion reference record exceeds 1048576 bytes".into(),
                    ));
                }
                pending.push(byte);
            }
        }
        if paths.len() > polylogue_store::claude_bodies::MAX_COMPANIONS {
            return Err(Error::ObservedSourceRefused(
                "Claude companion reference count exceeds 32".into(),
            ));
        }
    }
    for (index, (path, inline)) in paths.iter().enumerate() {
        let path = Path::new(path);
        let id = format!("{request}:claude-body:{index}");
        let copy = format!("copy:{}", path.display());
        writer.acquire_claude_body(
            &source.input.id,
            main,
            path,
            &ImportObservation {
                id: &id,
                machine,
                host,
                path: &copy,
                seen_at,
            },
            cancelled,
            *inline,
        )?;
    }
    writer.stored_source(&source.input.id)?.ok_or(Error::Range)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancelled_companion_scan_fails_even_for_an_empty_main() {
        let root = tempfile::tempdir().unwrap();
        let main = root.path().join("session.jsonl");
        std::fs::write(&main, b"").unwrap();
        let mut writer = Writer::open(&root.path().join("archive")).unwrap();
        let source = writer
            .acquire_claude_main(&main, "test", "test", "main", 0, &AtomicBool::new(false))
            .unwrap();
        assert!(matches!(
            acquire(
                &mut writer,
                &source,
                &main,
                "test",
                "test",
                "request",
                0,
                &AtomicBool::new(true)
            ),
            Err(Error::Cancelled)
        ));
    }
}
