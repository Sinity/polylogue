//! Worker-side configured acquisition. Parsing discovers only explicitly named
//! companions; the store prepares opaque proofs and the owner installs them.
use polylogue_model::BlobHash;
use polylogue_store::{BlobCapture, Error, RetainedSource, observed::PreparedClaude};
use std::{collections::BTreeMap, path::Path, sync::atomic::AtomicBool};
pub(crate) fn prepare(
    capture: BlobCapture,
    path: &Path,
    old: Option<RetainedSource>,
    cancelled: &AtomicBool,
    budget: polylogue_store::history_pages::turns::Budget,
    mut yield_turn: impl FnMut(
        polylogue_store::history_pages::turns::Progress,
    ) -> std::result::Result<(), Error>,
) -> std::result::Result<PreparedClaude, Error> {
    let state = std::cell::RefCell::new((
        Vec::new(),
        BTreeMap::<String, Option<(BlobHash, u64)>>::new(),
    ));
    capture.prepare_claude_in_turns(
        path,
        old,
        cancelled,
        budget,
        &mut yield_turn,
        |bytes| {
            let mut state = state.borrow_mut();
            let (pending, paths) = &mut *state;
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
        || {
            Ok(std::mem::take(&mut state.borrow_mut().1)
                .into_iter()
                .collect())
        },
    )
}
