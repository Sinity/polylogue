//! Input bytes as Sinex materials. Each step of a captured file's appended
//! bytes is one part of its real input: the origin that wrote it, its path
//! and host. A growing file's parts chain, each naming the part before it in
//! the same revision chain; a rewritten file starts a new chain. The
//! material's manifest holds the register rows that place the part
//! ([`InputRows`]), so a rebuild from Sinex puts every part back where it
//! was.
use crate::client::Client;
use crate::{Error, Result};
use polylogue_model::BlobHash;
use polylogue_store::input_rows::InputRows;
use polylogue_store::primary::{ConfirmedBlob, Held, Part, RegisterChange};
use sinex_model::{InputRef, MaterialId};
use sinex_protocol::{Ingress, MaterialFrame, ProducerMaterial};
use std::io::Read;

/// The parser name of register materials (looks that found no new bytes,
/// requests and excisions). An input's parts are read by
/// `polylogue-<origin>`.
pub(crate) const REGISTER: &str = "polylogue-register";
pub(crate) const ATTACHMENT: &str = "polylogue-attachment";

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
pub(crate) async fn send(
    client: &mut Client,
    part: &Part,
    bytes: &mut dyn Read,
) -> Result<ConfirmedBlob> {
    let extent = part
        .rows
        .extent
        .as_ref()
        .ok_or_else(|| Error::Invalid("part needs an extent".into()))?;
    if extent.blob != part.hash || extent.length < 0 || extent.length as u64 != part.length {
        return Err(Error::Invalid(
            "part rows differ from material bytes".into(),
        ));
    }
    let name = format!(
        "part:{}:{}:{}:{}",
        part.rows.input, extent.chain, extent.offset, part.hash
    );
    let blob = content_name(&part.hash);
    client.write_bytes(&name, &blob, part.length, bytes)?;
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
            manifest: sinex_model::Manifest {
                path: Some(part.input.path.clone()),
                details: [("polylogue".to_owned(), serde_json::to_value(&part.rows)?)]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
        }),
    };
    let ack = client.send(frame).await?;
    confirmed(ack, part.hash, part.length)
}

fn confirmed(
    ack: sinex_protocol::Acknowledgment,
    hash: BlobHash,
    length: u64,
) -> Result<ConfirmedBlob> {
    let id = ack
        .material
        .ok_or_else(|| Error::Invalid("acknowledgment has no material".into()))?;
    let confirmed = ConfirmedBlob {
        id: id.to_string(),
        extents: decode_extents(&ack.material_extents)?,
    };
    confirmed
        .extent(hash, length)
        .map_err(|error| Error::Invalid(error.to_string()))?;
    Ok(confirmed)
}

pub(crate) fn decode_extents(encoded: &[Vec<u8>]) -> Result<Vec<hashed_blobs::Extent>> {
    encoded
        .iter()
        .map(|bytes| hashed_blobs::Extent::decode(bytes).map_err(Error::from))
        .collect()
}

pub(crate) async fn send_blob(
    client: &mut Client,
    host: &str,
    hash: BlobHash,
    length: u64,
    bytes: &mut dyn Read,
) -> Result<ConfirmedBlob> {
    let name = format!("attachment:{hash}");
    let blob = content_name(&hash);
    client.write_bytes(&name, &blob, length, bytes)?;
    let ack = client
        .send(Ingress::Material {
            frame: name,
            material: Box::new(MaterialFrame {
                id: MaterialId::new(),
                input: InputRef {
                    kind: ATTACHMENT.into(),
                    locator: blob.clone(),
                    host: host.into(),
                },
                blob,
                length,
                parser: ATTACHMENT.into(),
                parser_version: parser_version(),
                part_index: 0,
                previous_part: None,
                manifest: sinex_model::Manifest {
                    details: [(
                        "polylogue_blob".into(),
                        serde_json::json!({"hash":hash.to_string(),"length":length}),
                    )]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                },
            }),
        })
        .await?;
    confirmed(ack, hash, length)
}

/// What one held material is to Polylogue: a part with the rows that place
/// it, or a register change read from its bytes in Sinex's content store.
pub(crate) fn held(store: &hashed_blobs::Reader, held: ProducerMaterial) -> Result<Held> {
    let extents = decode_extents(&held.extents)?;
    let material = held.material;
    let hash = parse_content_name(&material.blob)?;
    let confirmed = ConfirmedBlob {
        id: material.id.to_string(),
        extents: extents.clone(),
    };
    confirmed
        .extent(hash, material.length)
        .map_err(|error| Error::Invalid(error.to_string()))?;
    if material.parser == ATTACHMENT {
        let descriptor = material
            .manifest
            .details
            .get("polylogue_blob")
            .ok_or_else(|| Error::Invalid("attachment has no blob descriptor".into()))?;
        if descriptor.get("hash").and_then(serde_json::Value::as_str)
            != Some(hash.to_string().as_str())
            || descriptor.get("length").and_then(serde_json::Value::as_u64) != Some(material.length)
        {
            return Err(Error::Invalid(
                "attachment descriptor differs from material".into(),
            ));
        }
        return Ok(Held::Blob {
            number: held.number,
            material: material.id.to_string(),
            hash,
            length: material.length,
            extents,
        });
    }
    if material.parser == REGISTER {
        let bytes = store.read(&extents, None)?;
        if BlobHash::of(&bytes) != hash {
            return Err(Error::Invalid(format!(
                "{} failed its check",
                material.blob
            )));
        }
        let change: RegisterChange = serde_json::from_slice(&bytes)?;
        return Ok(Held::Change {
            number: held.number,
            material: material.id.to_string(),
            change,
        });
    }
    let rows: InputRows = serde_json::from_value(
        material
            .manifest
            .details
            .get("polylogue")
            .cloned()
            .ok_or_else(|| Error::Invalid(format!("{} has no register rows", held.frame)))?,
    )?;
    Ok(Held::Part {
        number: held.number,
        material: material.id.to_string(),
        hash,
        length: material.length,
        extents,
        index: material.part_index,
        rows: Box::new(rows),
    })
}
