//! Worker-side configured acquisition. Parsing discovers only explicitly named
//! companions; the store prepares opaque proofs and the owner installs them.
use polylogue_model::BlobHash;
use polylogue_store::{
    BlobCapture, Error, RetainedSource, claude_bodies::PreparedClaudeBody,
    observed::PreparedClaudeMain,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};
pub(crate) struct Prepared {
    pub main: PreparedClaudeMain,
    pub bodies: Vec<(PathBuf, PreparedClaudeBody)>,
}
pub(crate) fn prepare(
    capture: BlobCapture,
    path: &Path,
    old: Option<RetainedSource>,
    cancelled: &AtomicBool,
    budget: polylogue_store::history_pages::turns::Budget,
    mut yield_turn: impl FnMut(
        polylogue_store::history_pages::turns::Progress,
    ) -> std::result::Result<(), Error>,
) -> std::result::Result<Prepared, Error> {
    let mut pending = Vec::new();
    let mut paths: BTreeMap<String, Option<(BlobHash, u64)>> = BTreeMap::new();
    let main = capture.prepare_claude_main_in_turns(
        path,
        old,
        cancelled,
        budget,
        &mut yield_turn,
        |bytes| {
            for byte in bytes {
                if *byte == b'\n' {
                    if let Ok(record) = polylogue_model::RetainedJson::parse(&pending) {
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
                    pending.push(*byte);
                }
            }
            if paths.len() > polylogue_store::claude_bodies::MAX_COMPANIONS {
                return Err(Error::ObservedSourceRefused(
                    "Claude companion reference count exceeds 32".into(),
                ));
            }
            Ok(())
        },
    )?;
    let mut bodies = Vec::new();
    for (name, inline) in paths {
        let body = PathBuf::from(name);
        if let Some(prepared) = capture.prepare_claude_body_in_turns(
            path,
            &body,
            cancelled,
            inline,
            budget,
            &mut yield_turn,
        )? {
            bodies.push((body, prepared));
        }
    }
    Ok(Prepared { main, bodies })
}
