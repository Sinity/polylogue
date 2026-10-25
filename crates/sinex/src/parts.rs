//! Input bytes as Sinex materials. Each piece of a captured file is one part
//! of its real input: the origin that wrote it, its path and host. A growing
//! file's pieces chain, each naming the part before it in the same revision
//! chain; a rewritten file starts a new chain. The material's manifest holds
//! the register rows that place the piece ([`InputRecord`]), so a rebuild
//! from Sinex puts every piece back where it was.
use crate::client::Client;
use crate::{Error, Result};
use polylogue_model::BlobHash;
use polylogue_store::input_record::InputRecord;
use polylogue_store::primary::{Held, Part, Record};
use sinex_model::{InputRef, MaterialId};
use sinex_protocol::{HeldMaterial, Ingress, MaterialFrame};
use std::io::Read;

/// The parser name of record materials (requests, and looks that found no
/// new bytes). An input's parts are read by `polylogue-<origin>`.
pub(crate) const RECORDS: &str = "polylogue-records";

/// The version of what Polylogue's parsers make of the same bytes.
pub(crate) fn parser_version() -> u32 {
    polylogue_store::schema::INDEX_VERSION
}

/// `blake3:<hex>`, the form Sinex names bytes by.
pub(crate) fn content_name(hash: &BlobHash) -> String {
    format!("blake3:{hash}")
}

fn parse_content_name(name: &str) -> Result<BlobHash> {
    name.strip_prefix("blake3:")
        .and_then(|hex| BlobHash::parse(hex).ok())
        .ok_or_else(|| Error::Invalid(format!("{name} is not a BLAKE3 content name")))
}

/// Sends one part and returns the material Sinex registered. The frame is
/// named by where the bytes sit and what they are, so sending it again after
/// a stop is answered from Sinex's record.
pub(crate) async fn send(client: &mut Client, part: &Part, bytes: &mut dyn Read) -> Result<String> {
    let extent = part
        .record
        .extent
        .as_ref()
        .ok_or_else(|| Error::Invalid("a part needs an extent".into()))?;
    let name = format!(
        "part:{}:{}:{}:{}",
        part.record.input, extent.chain, extent.offset, part.hash
    );
    let blob = content_name(&part.hash);
    client.stage(&name, &blob, part.length, bytes)?;
    let previous = part
        .previous
        .as_deref()
        .map(|id| {
            id.parse()
                .map(MaterialId)
                .map_err(|e| Error::Invalid(format!("material {id}: {e}")))
        })
        .transpose()?;
    let frame = Ingress::Material {
        frame: name.clone(),
        material: Box::new(MaterialFrame {
            id: MaterialId::new(),
            input: InputRef {
                kind: part.input.origin.clone(),
                locator: part.input.path.clone(),
                host: part.input.host.clone(),
            },
            blob,
            length: part.length,
            parser: format!("polylogue-{}", part.input.origin),
            parser_version: parser_version(),
            part_index: part.index,
            previous_part: previous,
            manifest: serde_json::json!({ "polylogue": part.record }),
        }),
    };
    client
        .send(frame)
        .await?
        .material
        .map(|m| m.to_string())
        .ok_or_else(|| Error::Invalid(format!("{name} registered no material")))
}

/// What one held material is to Polylogue: a part with the rows that place
/// it, or a record read from its bytes in Sinex's content store.
pub(crate) fn held(store: &hashed_blobs::Reader, held: HeldMaterial) -> Result<Held> {
    let material = held.material;
    let hash = parse_content_name(&material.blob)?;
    if material.parser == RECORDS {
        let mut bytes = Vec::new();
        store
            .read(&hashed_blobs::Hash(hash.0))?
            .read_to_end(&mut bytes)?;
        if BlobHash::of(&bytes) != hash {
            return Err(Error::Invalid(format!("{} failed its check", material.blob)));
        }
        let record: Record = serde_json::from_slice(&bytes)?;
        return Ok(Held::Record {
            number: held.number,
            record,
        });
    }
    let record: InputRecord = serde_json::from_value(
        material
            .manifest
            .get("polylogue")
            .cloned()
            .ok_or_else(|| Error::Invalid(format!("{} has no register record", held.frame)))?,
    )?;
    Ok(Held::Part {
        number: held.number,
        material: material.id.to_string(),
        hash,
        length: material.length,
        index: material.part_index,
        record,
    })
}
