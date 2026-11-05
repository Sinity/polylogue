//! References into accepted pending operations. These indexes disappear with
//! custody and cannot assign an input revision, arrival, or parser cursor.
use super::*;
use polylogue_store::primary::SourceSelector;

#[derive(Default)]
pub(crate) struct Heads {
    pub head: Option<Prepared>,
    pub raw: Option<Prepared>,
    pub control: Option<Prepared>,
    pub unattributed: bool,
}

#[derive(Serialize, Deserialize)]
struct Pointer {
    order: u64,
    identity: String,
}

fn source_key(source: &SourceSelector) -> Result<String> {
    Ok(hash(&serde_json::to_vec(source)?))
}

fn latest(path: &Path, prepared: &Prepared) -> Result<()> {
    if let Some(previous) = read_json::<Pointer>(path)? {
        if previous.order > prepared.order {
            return Ok(());
        }
        if previous.order == prepared.order && previous.identity != prepared.identity {
            return invalid("pending source index repeats a custody order");
        }
    }
    write_derived(
        path,
        &Pointer {
            order: prepared.order,
            identity: prepared.identity.clone(),
        },
    )
}

pub(super) fn insert(spool: &Spool, prepared: &Prepared) -> Result<()> {
    if prepared.unresolved {
        return Ok(());
    }
    let Some(Capture::Input(input)) = prepared.capture() else {
        return Ok(());
    };
    let Some(basis) = &input.basis else {
        return write_index(&spool.index("pending-source", "unattributed"), "unknown");
    };
    let root = spool.index("pending-source", &source_key(&basis.source)?);
    fs::create_dir_all(root.join("control"))?;
    latest(&root.join("head"), prepared)?;
    if input.raw_range.is_some() {
        latest(&root.join("raw"), prepared)?;
    }
    if let Some(control) = prepared.control() {
        latest(
            &root
                .join("control")
                .join(hash(control.reading_version.as_bytes())),
            prepared,
        )?;
    }
    Ok(())
}

fn resolve(spool: &Spool, path: &Path, source: &SourceSelector) -> Result<Option<Prepared>> {
    let Some(pointer) = read_json::<Pointer>(path)? else {
        return Ok(None);
    };
    let prepared = spool.required(&pointer.identity)?;
    if prepared.order != pointer.order
        || !matches!(prepared.capture(), Some(Capture::Input(input)) if input.basis.as_ref().is_some_and(|basis| &basis.source == source))
    {
        return invalid("pending source index differs from its immutable operation");
    }
    Ok(Some(prepared))
}

pub(super) fn get(spool: &Spool, source: &SourceSelector, version: &str) -> Result<Heads> {
    let root = spool.index("pending-source", &source_key(source)?);
    let heads = Heads {
        head: resolve(spool, &root.join("head"), source)?,
        raw: resolve(spool, &root.join("raw"), source)?,
        control: resolve(
            spool,
            &root.join("control").join(hash(version.as_bytes())),
            source,
        )?,
        unattributed: spool.index("pending-source", "unattributed").try_exists()?,
    };
    if heads.raw.as_ref().is_some_and(
        |entry| !matches!(entry.capture(), Some(Capture::Input(input)) if input.raw_range.is_some()),
    ) || heads.control.as_ref().is_some_and(|entry| {
        entry
            .control()
            .is_none_or(|control| control.reading_version != version)
    }) {
        return invalid("pending source index points to another capture role");
    }
    Ok(heads)
}
