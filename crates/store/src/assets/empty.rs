//! Empty members are backed by an exact receiving observation and encoded body.
use super::*;
use crate::primary::{AcquisitionReceipt, AcquisitionRoute, RawRange};
use polylogue_model::BlobHash;

pub(crate) fn source(
    index: &Connection,
    conn: &Connection,
    prefix: &str,
    input: &str,
    chain: i64,
    observation: Option<&str>,
    version: Option<&str>,
) -> Result<Option<AssetSource>> {
    if input.is_empty() || chain <= 0 {
        return Ok(None);
    }
    let mut query = conn.prepare(&format!(
        "SELECT o.id,o.machine,o.path_bytes,o.details_json,i.format
         FROM {prefix}observation o JOIN {prefix}input i ON i.id=o.input
         LEFT JOIN {prefix}revision r ON r.input=o.input AND r.number=o.revision
         WHERE o.input=?1 AND o.commit_seq>0 AND coalesce(o.chain,r.chain)=?2 AND coalesce(o.length,r.length)=0
           AND (?3 IS NULL OR o.id=?3)
           AND EXISTS(SELECT 1 FROM {prefix}revision v WHERE v.input=o.input AND v.chain=?2 AND v.length=0 AND v.commit_seq>0)
         ORDER BY o.commit_seq DESC,o.rowid DESC"
    ))?;
    let mut rows = query.query(params![input, chain, observation])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let machine: String = row.get(1)?;
        let path: Option<Vec<u8>> = row.get(2)?;
        let details: Option<String> = row.get(3)?;
        let format: String = row.get(4)?;
        let membership: Option<(String, String)> = index
            .query_row(
                "SELECT import_id,member FROM import_member WHERE observation=?1 AND input=?2",
                params![id, input],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let Some((namespace, member)) = membership else {
            continue;
        };
        let (Some(path), Some(details)) = (path, details) else {
            continue;
        };
        let details: serde_json::Value = serde_json::from_str(&details)?;
        let (Some(import), Some(acquisition)) = (details.get("import"), details.get("acquisition"))
        else {
            continue;
        };
        let import: ImportMember = serde_json::from_value(import.clone())?;
        let receipt: AcquisitionReceipt = serde_json::from_value(acquisition.clone())?;
        let key = crate::InputKey {
            id: input.to_owned(),
            format: format.clone(),
            route: polylogue_model::Route::Export,
        };
        if import.version != 1
            || import.import_id.is_empty()
            || import.member.is_empty()
            || import.import_id != namespace
            || import.member != member
            || format.is_empty()
            || machine.is_empty()
            || path.is_empty()
            || receipt.route != AcquisitionRoute::ExplicitImport
            || receipt.source.origin != format
            || receipt.source.machine != machine
            || receipt.source.path_bytes != path
            || receipt.range != (RawRange { start: 0, end: 0 })
            || receipt.hash != BlobHash::of(b"")
        {
            continue;
        }
        receipt.validate_for(&key)?;
        let descriptor: Option<Vec<u8>> = conn
            .query_row(
                &format!("SELECT storage FROM {prefix}blob WHERE hash=?1 AND size=0"),
                [&receipt.hash.0[..]],
                |row| row.get(0),
            )
            .optional()?;
        let Some(descriptor) = descriptor else {
            continue;
        };
        let extent = hashed_blobs::Extent::decode(&descriptor)?;
        if extent.hash.0 != receipt.hash.0
            || extent.decoded_len != 0
            || extent.len == 0
            || extent.kind != hashed_blobs::Kind::Plain
        {
            return Err(StoreError::BlobCorrupt(receipt.hash));
        }
        let mut digest = blake3::Hasher::new();
        digest.update(b"polylogue.empty-import-member.v1\0");
        digest.update(&serde_json::to_vec(&(
            input, chain, &id, &namespace, &member, &receipt,
        ))?);
        let actual = digest.finalize().to_hex().to_string();
        if version.is_some_and(|expected| expected != actual) {
            continue;
        }
        return Ok(Some(AssetSource {
            chain,
            length: 0,
            version: actual,
            blob: Some(receipt.hash),
        }));
    }
    Ok(None)
}

/// A zero decoded length still opens and checks every committed frame. A
/// zero-sized range on StoredChain cannot provide that physical verification.
pub(crate) fn read(reader: &Reader<'_>, source: &AssetSource) -> Result<Vec<u8>> {
    let hash = source
        .blob
        .filter(|hash| source.length == 0 && *hash == BlobHash::of(b""))
        .ok_or_else(|| StoreError::Invalid("empty attachment has no bound body".into()))?;
    let extent = crate::blobs::load_extent(reader.conn()?, "source_", &hashed_blobs::Hash(hash.0))?
        .ok_or_else(|| StoreError::NotFound("empty attachment body".into()))?;
    if extent.decoded_len != 0
        || extent.hash.0 != hash.0
        || extent.kind != hashed_blobs::Kind::Plain
    {
        return Err(StoreError::BlobCorrupt(hash));
    }
    let store = reader.blob_reader()?;
    let codec = hashed_blobs::Reader::new(store.root());
    let mismatches = codec.scan_checked(std::slice::from_ref(&extent), || {
        reader
            .conn()
            .map(|_| ())
            .map_err(|error| hashed_blobs::Error::Io(std::io::Error::other(error)))
    })?;
    if !mismatches.is_empty() {
        return Err(StoreError::BlobCorrupt(hash));
    }
    let bytes = codec.read(&[extent], None)?;
    reader.conn()?;
    if !bytes.is_empty() || BlobHash::of(&bytes) != hash {
        return Err(StoreError::BlobCorrupt(hash));
    }
    Ok(bytes)
}
