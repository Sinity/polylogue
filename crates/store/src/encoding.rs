//! Encoding of database row bodies. Original-input blob encoding belongs to `blobs`.

use crate::{Result, StoreError};
use polylogue_model::{
    Block, Document, DocumentId, Event, Message, MessageId, Timestamp, bundle::ShownBlocks,
};
use serde::{Serialize, de::DeserializeOwned};
use std::cell::RefCell;
use std::io::Read;
use std::sync::LazyLock;

// Fast row-body compression; the format remains readable across compression levels.
const LEVEL: i32 = 1;

// Fixed for one index generation. Archive rows use pack_bytes
// below; only RowPacker may introduce dictionary frames.
const ROW_DICTIONARY: &[u8] = include_bytes!("encoding/rows.dict");
static ROW_ENCODER: LazyLock<zstd::dict::EncoderDictionary<'static>> =
    LazyLock::new(|| zstd::dict::EncoderDictionary::copy(ROW_DICTIONARY, LEVEL));
static ROW_DECODER: LazyLock<zstd::dict::DecoderDictionary<'static>> =
    LazyLock::new(|| zstd::dict::DecoderDictionary::copy(ROW_DICTIONARY));

// Codec contexts are reused by each thread: creating one per call cost more
// than packing a typical row.
thread_local! {
    static PLAIN_PACKER: RefCell<Option<zstd::bulk::Compressor<'static>>> =
        const { RefCell::new(None) };
    static PLAIN_UNPACKER: RefCell<Option<zstd::bulk::Decompressor<'static>>> =
        const { RefCell::new(None) };
    static ROW_UNPACKER: RefCell<Option<zstd::bulk::Decompressor<'static>>> =
        const { RefCell::new(None) };
}

pub(crate) fn pack_bytes(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    PLAIN_PACKER.with(|packer| {
        let mut packer = packer.borrow_mut();
        let packer = match &mut *packer {
            Some(packer) => packer,
            empty => empty.insert(zstd::bulk::Compressor::new(LEVEL)?),
        };
        packer.compress(bytes)
    })
}

pub(crate) fn unpack_bytes(packed: &[u8]) -> std::io::Result<Vec<u8>> {
    let dictionary = zstd::zstd_safe::get_dict_id_from_frame(packed).is_some();
    // A frame states its decoded size when it was packed whole; anything
    // else, and a size that does not hold, decodes through the stream path.
    if let Ok(Some(size)) = zstd::zstd_safe::get_frame_content_size(packed)
        && let Ok(size) = usize::try_from(size)
    {
        let unpacker = if dictionary {
            &ROW_UNPACKER
        } else {
            &PLAIN_UNPACKER
        };
        let unpacked = unpacker.with(|unpacker| {
            let mut unpacker = unpacker.borrow_mut();
            let unpacker = match &mut *unpacker {
                Some(unpacker) => unpacker,
                empty => empty.insert(if dictionary {
                    zstd::bulk::Decompressor::with_prepared_dictionary(&ROW_DECODER)?
                } else {
                    zstd::bulk::Decompressor::new()?
                }),
            };
            unpacker.decompress(packed, size)
        });
        if let Ok(bytes) = unpacked
            && bytes.len() == size
        {
            return Ok(bytes);
        }
    }
    if !dictionary {
        return zstd::stream::decode_all(packed);
    }
    let mut decoder = zstd::stream::read::Decoder::with_prepared_dictionary(packed, &ROW_DECODER)?;
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub(crate) fn pack_text(text: &str) -> std::io::Result<Vec<u8>> {
    pack_bytes(text.as_bytes())
}

pub(crate) fn normalized_text_hash(text: &str) -> [u8; 32] {
    use unicode_normalization::UnicodeNormalization;
    if unicode_normalization::is_nfc(text) {
        *blake3::hash(text.as_bytes()).as_bytes()
    } else {
        *blake3::hash(text.nfc().collect::<String>().as_bytes()).as_bytes()
    }
}

pub(crate) fn unpack_text(packed: &[u8]) -> Result<String> {
    String::from_utf8(unpack_bytes(packed)?).map_err(StoreError::stored_data)
}

/// Packs source metadata without the index row dictionary.
pub(crate) fn pack_json(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(pack_bytes(&serde_json::to_vec(value)?)?)
}

/// Source time; SQL milliseconds are only query caches.
pub(crate) fn time_bytes(
    at: Option<Timestamp>,
    accuracy: polylogue_model::TimeAccuracy,
) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(at, accuracy))?)
}

/// A stored value the model cannot read back.
#[derive(Debug, thiserror::Error)]
enum Unreadable {
    #[error("unknown stored name {0:?}")]
    Name(String),
    #[error("stored identity {0:?} does not parse: {1}")]
    Identity(String, String),
    #[error("a stored hash has {0} bytes, not 32")]
    Hash(usize),
    #[error("stored time {0} ms is out of range")]
    Time(i64),
}

fn unreadable(column: usize, kind: rusqlite::types::Type, error: Unreadable) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(column, kind, Box::new(error))
}

/// An identity the archive stores as its text form: an ID or an author.
pub(crate) trait StoredId: Sized {
    fn parse_stored(text: &str) -> std::result::Result<Self, String>;
}

macro_rules! stored_id {
    ($($ty:ty),+) => {$(
        impl StoredId for $ty {
            fn parse_stored(text: &str) -> std::result::Result<Self, String> {
                <$ty>::parse(text).map_err(|error| error.to_string())
            }
        }
    )+};
}

stored_id!(
    polylogue_model::SessionId,
    MessageId,
    DocumentId,
    polylogue_model::RunId,
    polylogue_model::Author
);

/// Typed decoding of index columns. A stored value the model cannot read back
/// is a column error naming the column, never a default, a skipped row or a
/// silently missing value. Unsigned counts need none of this: rusqlite's
/// `fallible_uint` refuses a negative stored count and an unsigned value above
/// `i64::MAX` on write.
pub(crate) trait Columns {
    fn name<T: polylogue_model::StoredName>(&self, column: usize) -> rusqlite::Result<T>;
    fn optional_name<T: polylogue_model::StoredName>(
        &self,
        column: usize,
    ) -> rusqlite::Result<Option<T>>;
    fn id<T: StoredId>(&self, column: usize) -> rusqlite::Result<T>;
    fn optional_id<T: StoredId>(&self, column: usize) -> rusqlite::Result<Option<T>>;
    fn hash(&self, column: usize) -> rusqlite::Result<[u8; 32]>;
    fn optional_hash(&self, column: usize) -> rusqlite::Result<Option<[u8; 32]>>;
    fn time_ms(&self, column: usize) -> rusqlite::Result<Timestamp>;
    fn optional_time_ms(&self, column: usize) -> rusqlite::Result<Option<Timestamp>>;
}

fn decode_name<T: polylogue_model::StoredName>(column: usize, name: String) -> rusqlite::Result<T> {
    T::from_name(&name)
        .ok_or_else(|| unreadable(column, rusqlite::types::Type::Text, Unreadable::Name(name)))
}

fn decode_id<T: StoredId>(column: usize, text: String) -> rusqlite::Result<T> {
    T::parse_stored(&text).map_err(|error| {
        unreadable(
            column,
            rusqlite::types::Type::Text,
            Unreadable::Identity(text, error),
        )
    })
}

fn decode_hash(column: usize, bytes: Vec<u8>) -> rusqlite::Result<[u8; 32]> {
    let len = bytes.len();
    bytes
        .try_into()
        .map_err(|_| unreadable(column, rusqlite::types::Type::Blob, Unreadable::Hash(len)))
}

fn decode_time(column: usize, ms: i64) -> rusqlite::Result<Timestamp> {
    Timestamp::from_unix_ms(ms)
        .ok_or_else(|| unreadable(column, rusqlite::types::Type::Integer, Unreadable::Time(ms)))
}

impl Columns for rusqlite::Row<'_> {
    fn name<T: polylogue_model::StoredName>(&self, column: usize) -> rusqlite::Result<T> {
        decode_name(column, self.get(column)?)
    }
    fn optional_name<T: polylogue_model::StoredName>(
        &self,
        column: usize,
    ) -> rusqlite::Result<Option<T>> {
        self.get::<_, Option<String>>(column)?
            .map(|name| decode_name(column, name))
            .transpose()
    }
    fn id<T: StoredId>(&self, column: usize) -> rusqlite::Result<T> {
        decode_id(column, self.get(column)?)
    }
    fn optional_id<T: StoredId>(&self, column: usize) -> rusqlite::Result<Option<T>> {
        self.get::<_, Option<String>>(column)?
            .map(|text| decode_id(column, text))
            .transpose()
    }
    fn hash(&self, column: usize) -> rusqlite::Result<[u8; 32]> {
        decode_hash(column, self.get(column)?)
    }
    fn optional_hash(&self, column: usize) -> rusqlite::Result<Option<[u8; 32]>> {
        self.get::<_, Option<Vec<u8>>>(column)?
            .map(|bytes| decode_hash(column, bytes))
            .transpose()
    }
    fn time_ms(&self, column: usize) -> rusqlite::Result<Timestamp> {
        decode_time(column, self.get(column)?)
    }
    fn optional_time_ms(&self, column: usize) -> rusqlite::Result<Option<Timestamp>> {
        self.get::<_, Option<i64>>(column)?
            .map(|ms| decode_time(column, ms))
            .transpose()
    }
}

pub(crate) fn read_time(
    row: &rusqlite::Row<'_>,
    column: usize,
) -> rusqlite::Result<(Option<Timestamp>, polylogue_model::TimeAccuracy)> {
    let bytes: Vec<u8> = row.get(column)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            rusqlite::types::Type::Blob,
            Box::new(error),
        )
    })
}

/// One message's identity and content, prepared together before its parsed
/// payloads are released. Its content lies in the content groups prepared
/// with it ([`message_content`]); the writer stores those groups and names
/// the piece, without encoding anything again.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedMessage {
    pub(crate) id: MessageId,
    pub(crate) block_selection: ShownBlocks,
    pub(crate) time: Vec<u8>,
    /// The message's content hash: its version.
    pub(crate) hash: [u8; 32],
    /// Each block's text hash, for a block that keeps its text in `text`.
    pub(crate) texts: Vec<Option<[u8; 32]>>,
    pub(crate) content: crate::content::Placed,
}

impl PreparedMessage {
    /// Prepares a message holding every block of its version, laying its
    /// content into `packer`.
    pub(crate) fn new(
        message: &mut Message,
        stored: &[polylogue_model::StoredField],
        packer: &mut crate::content::Packer,
    ) -> Result<Self> {
        let block_selection = ShownBlocks {
            source_version: message.content_hash()?,
            block_count: message.blocks.len() as u64,
            block_ordinals: (0..message.blocks.len() as u64).collect(),
        };
        Self::of(message, block_selection, stored, packer)
    }

    /// Prepares a message holding the blocks `block_selection` names of its
    /// version.
    pub(crate) fn projected(
        message: &mut Message,
        block_selection: &ShownBlocks,
        stored: &[polylogue_model::StoredField],
        packer: &mut crate::content::Packer,
    ) -> Result<Self> {
        block_selection.validate(message.blocks.len())?;
        if block_selection.block_count == message.blocks.len() as u64
            && message.content_hash()? != block_selection.source_version
        {
            return Err(polylogue_model::bundle::BundleError::SourceVersionMismatch.into());
        }
        Self::of(message, block_selection.clone(), stored, packer)
    }

    fn of(
        message: &mut Message,
        block_selection: ShownBlocks,
        stored: &[polylogue_model::StoredField],
        packer: &mut crate::content::Packer,
    ) -> Result<Self> {
        let content = packer.add(&message_content(message, stored)?)?;
        Ok(Self {
            id: message.id.clone(),
            time: time_bytes(message.at, message.time_accuracy())?,
            hash: block_selection.source_version.0,
            texts: text_hashes(&message.blocks),
            block_selection,
            content,
        })
    }
}

/// Each block's text hash, for a block that keeps its text in `text`.
pub(crate) fn text_hashes(blocks: &[Block]) -> Vec<Option<[u8; 32]>> {
    blocks
        .iter()
        .map(|block| block_text(block).map(|text| *blake3::hash(text.as_bytes()).as_bytes()))
        .collect()
}

/// A message's content as stored once (docs/archive.md, content): its fields
/// without blocks, then each block with its text moved out and the block's
/// structured value apart from it (empty when it has none), as the parts of
/// one piece ([`crate::content::join_parts`]). A value the parser left in
/// the stored bytes (`stored`) is its marker ([`crate::content::stored_marker`]).
/// The message is given back as it was.
pub(crate) fn message_content(
    message: &mut Message,
    stored: &[polylogue_model::StoredField],
) -> Result<Vec<u8>> {
    let blocks = std::mem::take(&mut message.blocks);
    let header = serde_json::to_vec(&*message);
    message.blocks = blocks;
    let mut parts = vec![header?];
    for (n, block) in message.blocks.iter().enumerate() {
        let left: Vec<_> = stored
            .iter()
            .filter(|field| field.block == Some(n as u64))
            .collect();
        if left.is_empty() {
            parts.push(block_bytes(block)?);
        } else {
            let mut body: serde_json::Value = serde_json::from_slice(&block_bytes(block)?)?;
            for field in left {
                crate::content::leave_stored(&mut body, field)?;
            }
            parts.push(serde_json::to_vec(&body)?);
        }
        parts.push(
            block_data(block)
                .map(serde_json::to_vec)
                .transpose()?
                .unwrap_or_default(),
        );
    }
    let parts: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    Ok(crate::content::join_parts(&parts))
}

/// Where one block's piece lies in its message's content: its body, then
/// its structured value when it has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockPiece {
    pub(crate) start: usize,
    pub(crate) length: usize,
    pub(crate) data_length: Option<usize>,
}

/// A message's content read back ([`message_content`]): the message with its
/// texts still moved out, its blocks' structured values restored when `data`,
/// and where each block's piece lies in the content.
pub(crate) fn read_message_content(
    content: &[u8],
    data: bool,
) -> Result<(Message, Vec<BlockPiece>)> {
    let parts = crate::content::parts(content)?;
    if parts.len() % 2 != 1 {
        return Err(StoreError::stored(0, "a message's content has no fields"));
    }
    let mut message: Message = read_json(&content[parts[0].clone()])?;
    let mut pieces = Vec::with_capacity(parts.len() / 2);
    message.blocks = Vec::with_capacity(parts.len() / 2);
    for pair in parts[1..].chunks(2) {
        let (body, value) = (pair[0].clone(), pair[1].clone());
        let mut block: Block = read_json(&content[body.clone()])?;
        if data && !value.is_empty() {
            set_data(&mut block, read_json(&content[value.clone()])?);
        }
        message.blocks.push(block);
        pieces.push(BlockPiece {
            start: body.start,
            length: body.len(),
            data_length: (!value.is_empty()).then_some(value.len()),
        });
    }
    Ok((message, pieces))
}

/// One event's content and value hash, prepared before its parsed payload is
/// released ([`event_content`]).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedEvent {
    /// The ID a session's event is named by when its source gives none.
    pub(crate) content_id: String,
    /// The hash of its data, by which composition chooses between readings.
    pub(crate) hash: [u8; 32],
    /// Its fields then its data.
    pub(crate) content: crate::content::Placed,
    pub(crate) fields_length: u64,
}

impl PreparedEvent {
    pub(crate) fn new(
        event: &mut Event,
        stored: &[polylogue_model::StoredField],
        packer: &mut crate::content::Packer,
    ) -> Result<Self> {
        let (content_id, mut data) = event_bytes(event)?;
        let hash = value_hash(&event.data)?;
        if !stored.is_empty() {
            let mut value = event.data.clone();
            for field in stored {
                crate::content::leave_stored(&mut value, field)?;
            }
            data = serde_json::to_vec(&value)?;
        }
        let fields = event_fields(event)?;
        let mut piece = fields;
        let fields_length = piece.len() as u64;
        piece.extend_from_slice(&data);
        Ok(Self {
            content_id,
            hash,
            content: packer.add(&piece)?,
            fields_length,
        })
    }

    /// The length of its data, which follows its fields.
    pub(crate) fn data_length(&self) -> u64 {
        self.content.length - self.fields_length
    }
}

/// An event's fields as stored: the encoded Event without its data. The
/// event is given back as it was.
pub(crate) fn event_fields(event: &mut Event) -> Result<Vec<u8>> {
    let data = std::mem::take(&mut event.data);
    let fields = serde_json::to_vec(&*event);
    event.data = data;
    Ok(fields?)
}

/// The hash a value is chosen by between readings: of its canonical JSON.
pub(crate) fn value_hash(value: &impl serde::Serialize) -> Result<[u8; 32]> {
    let canonical = canonical_json::canonical_json(value).map_err(identity_encoding)?;
    Ok(*blake3::hash(&canonical).as_bytes())
}

/// A document's block rows, packed: each block's body, its text's hash and
/// its structured value.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PackedBlocks {
    pub(crate) hash: [u8; 32],
    pub(crate) blocks: Vec<Vec<u8>>,
    pub(crate) texts: Vec<Option<[u8; 32]>>,
    /// Each block's structured value, packed for `document_block.data`.
    pub(crate) data: Vec<Option<Vec<u8>>>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedDocument {
    pub(crate) id: DocumentId,
    pub(crate) time: Vec<u8>,
    /// The document without its blocks, packed JSON.
    pub(crate) fields: Vec<u8>,
    pub(crate) body: Vec<u8>,
    content: PackedBlocks,
}

impl std::ops::Deref for PreparedDocument {
    type Target = PackedBlocks;
    fn deref(&self) -> &Self::Target {
        &self.content
    }
}

impl PreparedDocument {
    pub(crate) fn new(document: &Document) -> Result<Self> {
        RowPacker::new()?.pack_document(document)
    }
}

pub(crate) fn read_json<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let mut reader = serde_json::Deserializer::from_slice(bytes);
    reader.disable_recursion_limit();
    let value = T::deserialize(&mut reader)?;
    reader.end()?;
    Ok(value)
}

pub(crate) fn unpack_json<T: DeserializeOwned>(packed: &[u8]) -> Result<T> {
    read_json(&unpack_bytes(packed)?)
}

/// One independent frame per row; both encoders are reused for the batch.
pub(crate) struct RowPacker {
    plain: zstd::bulk::Compressor<'static>,
    rows: zstd::bulk::Compressor<'static>,
    empty_text_block: Option<Vec<u8>>,
}

impl RowPacker {
    pub(crate) fn new() -> std::io::Result<Self> {
        Ok(Self {
            plain: zstd::bulk::Compressor::new(LEVEL)?,
            rows: zstd::bulk::Compressor::with_prepared_dictionary(&ROW_ENCODER)?,
            empty_text_block: None,
        })
    }

    fn pack_row(&mut self, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        let plain = self.plain.compress(bytes)?;
        let dictionary = self.rows.compress(bytes)?;
        Ok(if dictionary.len() < plain.len() {
            dictionary
        } else {
            plain
        })
    }

    pub(crate) fn pack_json(&mut self, value: &impl Serialize) -> Result<Vec<u8>> {
        Ok(self.pack_row(&serde_json::to_vec(value)?)?)
    }

    pub(crate) fn pack_document(&mut self, document: &Document) -> Result<PreparedDocument> {
        let fields = Document {
            id: document.id.clone(),
            purpose: document.purpose,
            title: document.title.clone(),
            source_time: document.source_time,
            source_time_quality: document.source_time_quality,
            source_time_precision: document.source_time_precision,
            source_time_until: document.source_time_until,
            authorship: document.authorship,
            blocks: Vec::new(),
            metadata: document.metadata.clone(),
        };
        let hash = *canonical_json::canonical_hash("polylogue-document-v1", document)
            .map_err(identity_encoding)?
            .as_bytes();
        Ok(PreparedDocument {
            id: document.id.clone(),
            time: time_bytes(document.source_time, document.time_accuracy())?,
            body: self.pack_version(document, &document.blocks)?,
            fields: self.pack_json(&fields)?,
            content: self.pack_blocks(&document.blocks, hash)?,
        })
    }

    fn pack_version(&mut self, value: &impl Serialize, blocks: &[Block]) -> Result<Vec<u8>> {
        let mut value = serde_json::to_value(value)?;
        for (stored, block) in value["blocks"]
            .as_array_mut()
            .expect("blocks are an array")
            .iter_mut()
            .zip(blocks)
        {
            if block_text(block).is_some() {
                let field = match block {
                    Block::ToolResult { .. } => "output",
                    Block::ToolCall { .. } => "written_text",
                    _ => "text",
                };
                stored[field] = serde_json::Value::String(String::new());
            }
        }
        self.pack_json(&value)
    }

    fn pack_block_body(&mut self, block: &Block) -> Result<Vec<u8>> {
        if matches!(block, Block::Text { .. }) {
            // The content is stored in text groups. Every prose block has the
            // same packed body, regardless of its original text.
            if let Some(body) = &self.empty_text_block {
                return Ok(body.clone());
            }
            let body = self.pack_row(&block_bytes(block)?)?;
            self.empty_text_block = Some(body.clone());
            return Ok(body);
        }
        Ok(self.pack_row(&block_bytes(block)?)?)
    }

    fn pack_blocks(&mut self, blocks: &[Block], hash: [u8; 32]) -> Result<PackedBlocks> {
        let mut packed = Vec::with_capacity(blocks.len());
        let mut data = Vec::with_capacity(blocks.len());
        for block in blocks {
            packed.push(self.pack_block_body(block)?);
            data.push(
                block_data(block)
                    .map(|value| self.pack_json(value))
                    .transpose()?,
            );
        }
        Ok(PackedBlocks {
            hash,
            texts: text_hashes(blocks),
            blocks: packed,
            data,
        })
    }
}

/// One-shot block encoding for compatibility checks.
#[cfg(test)]
pub(crate) fn pack_block(block: &Block) -> Result<Vec<u8>> {
    Ok(pack_bytes(&block_bytes(block)?)?)
}

fn block_bytes(block: &Block) -> Result<Vec<u8>> {
    let body = match block {
        Block::Text { .. } => Block::Text {
            text: String::new(),
        },
        Block::Thinking {
            text: Some(_),
            signature,
            redacted,
        } => Block::Thinking {
            text: Some(String::new()),
            signature: signature.clone(),
            redacted: *redacted,
        },
        Block::ToolResult {
            reported_duration_ms,
            call_id,
            outcome,
            output,
            data,
            attachments,
        } if output.is_some() || data.is_some() => Block::ToolResult {
            reported_duration_ms: *reported_duration_ms,
            call_id: call_id.clone(),
            outcome: outcome.clone(),
            output: output.as_ref().map(|_| String::new()),
            data: None,
            attachments: attachments.clone(),
        },
        Block::ToolCall {
            written_text: Some(_),
            ..
        } => {
            let mut body = block.clone();
            if let Block::ToolCall { written_text, .. } = &mut body {
                *written_text = Some(String::new());
            }
            body
        }
        // These values are stored whole, so serialization can borrow them.
        _ => return Ok(serde_json::to_vec(block)?),
    };
    Ok(serde_json::to_vec(&body)?)
}

pub(crate) fn identity_encoding(error: canonical_json::Error) -> StoreError {
    StoreError::Invalid(error.to_string())
}

/// The ID an event of a session is named by when its source gives none: its
/// session, kind, time, data and the message it follows, so identical state
/// recorded after different messages is distinct. Byte ranges do not change
/// the event's identity when another input supplies it.
pub(crate) fn event_content_id(e: &Event) -> Result<String> {
    Ok(event_bytes(e)?.0)
}

fn event_bytes(e: &Event) -> Result<(String, Vec<u8>)> {
    let data = serde_json::to_vec(&e.data)?;
    if let Some(id) = reading_id(e) {
        return Ok((id, data));
    }
    let mut h = blake3::Hasher::new();
    if let Some(session) = &e.session {
        h.update(session.to_string().as_bytes());
    } else {
        h.update(b"\0input-event-content-v1\0");
    }
    h.update(b"\0");
    h.update(e.kind.as_bytes());
    h.update(b"\0");
    h.update(
        &e.at
            .map(Timestamp::unix_ms)
            .unwrap_or(i64::MIN)
            .to_le_bytes(),
    );
    h.update(&canonical_json::canonical_json(&e.data).map_err(identity_encoding)?);
    if let Some(after) = &e.after {
        h.update(b"\0after\0");
        h.update(after.to_string().as_bytes());
    }
    Ok((h.finalize().to_hex().to_string(), data))
}

/// A usage or quota reading names its own identity, which every copy of
/// it shares: it is stored once per origin under that identity.
fn reading_id(e: &Event) -> Option<String> {
    use polylogue_model::reports::{QUOTA_REPORT, USAGE_REPORT};
    if e.kind != USAGE_REPORT && e.kind != QUOTA_REPORT {
        return None;
    }
    let reading = e.data.get("report_id")?.as_str()?;
    let origin = e.session.as_ref()?.origin();
    let mut h = blake3::Hasher::new_derive_key("polylogue reading v1");
    for part in [e.kind.as_str(), origin.as_str(), reading] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part.as_bytes());
    }
    Some(h.finalize().to_hex().to_string())
}

/// Reads a block back from a packed `document_block.body`.
pub(crate) fn unpack_block(packed: &[u8]) -> Result<Block> {
    unpack_json(packed)
}

/// The text a block keeps in the `text` table rather than its packed body.
pub(crate) fn block_text(block: &Block) -> Option<&str> {
    match block {
        Block::Text { text } if !text.is_empty() => Some(text),
        Block::Thinking { text: Some(t), .. } if !t.is_empty() => Some(t),
        Block::ToolCall {
            written_text: Some(t),
            ..
        } if !t.is_empty() => Some(t),
        Block::ToolResult {
            output: Some(o), ..
        } if !o.is_empty() => Some(o),
        _ => None,
    }
}

/// The structured value a block keeps apart from its body: a tool result's
/// data. No text view shows it, and it can be most of a large message, so a
/// text read never decodes it.
pub(crate) fn block_data(block: &Block) -> Option<&serde_json::Value> {
    match block {
        Block::ToolResult {
            data: Some(data), ..
        } => Some(data),
        _ => None,
    }
}

/// Puts a block's structured value back.
pub(crate) fn set_data(block: &mut Block, value: serde_json::Value) {
    if let Block::ToolResult { data, .. } = block {
        *data = Some(value);
    }
}

/// Puts a block's text back from its `text` row.
pub(crate) fn restore_text(block: &mut Block, text: String) {
    match block {
        Block::Text { text: t } => *t = text,
        Block::Thinking { text: t, .. } => *t = Some(text),
        Block::ToolResult { output, .. } => *output = Some(text),
        Block::ToolCall { written_text, .. } => *written_text = Some(text),
        _ => {}
    }
}

/// A message's place in its session (docs/sessions.md, messages), written so
/// that comparing the text compares (order time, input ID, sequence, record
/// position): `0` for an unknown order time or `1` and the time, then the
/// input ID ended by U+0001 (below every character an ID holds, so a shorter
/// ID sorts before a longer one it begins), then the sequence and the record
/// position. Each number is sixteen lowercase hexadecimal digits of its value
/// with the sign bit flipped. Text, not bytes, because query pages carry their
/// sort keys as text.
pub(crate) fn session_order(
    order_time: Option<i64>,
    input: &str,
    seq: i64,
    position: i64,
) -> String {
    let sortable = |n: i64| format!("{:016x}", (n as u64) ^ (1 << 63));
    let time = match order_time {
        None => "0".to_owned(),
        Some(ms) => format!("1{}", sortable(ms)),
    };
    format!("{time}{input}\u{1}{}{}", sortable(seq), sortable(position))
}

#[cfg(test)]
mod tests;

/// A message reading's block layout: the original block count and the
/// ordinals of the blocks it holds.
pub(crate) fn message_layout(block_selection: &ShownBlocks) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(
        block_selection.block_count,
        &block_selection.block_ordinals,
    ))?)
}

/// The block selection a stored layout states for version `hash`; no layout
/// means every one of `blocks` blocks.
pub(crate) fn message_block_selection(
    hash: &[u8],
    layout: Option<&[u8]>,
    blocks: usize,
) -> Result<ShownBlocks> {
    let source_version = polylogue_model::BlobHash(hash.try_into().map_err(|_| {
        crate::StoreError::Invalid("a message source version is not a BLAKE3 hash".into())
    })?);
    let (block_count, block_ordinals) = match layout {
        Some(layout) => read_json::<(u64, Vec<u64>)>(layout)?,
        None => (blocks as u64, (0..blocks as u64).collect()),
    };
    let block_selection = ShownBlocks {
        source_version,
        block_count,
        block_ordinals,
    };
    block_selection.validate(block_selection.block_ordinals.len())?;
    Ok(block_selection)
}
