//! Shared fixture for the Message preparation tests: a temporary archive with
//! one stored input, a started parser and one synthetic Message identity.
use super::*;
use crate::contributions::{
    BlockKind, BlockPresence, BodyRetirement, CandidateCollection, CompatibilityTurn,
    CompatibleSelection, FieldRead, MessageLayout, MessageLookup, PreparedProjection,
    ProjectionStep, Retirement, SelectCompatibleSourceForTest, SmallField, SmallFieldEntry,
    SourceProjection, SourceRange, SourceRangeEntry, SourceSelection, SourceStep, ValueCollection,
    directory as d,
};
use crate::{ImportObservation, InputFormat};
use polylogue_model::merge::{BlockField, FinishState, MessageField, Slot};
use polylogue_model::{BlobHash, BlockSlot};

mod blocks;
mod compatibility;
mod fields;
mod inheritance;
mod inspection;
mod native_identity;
mod preparation;
mod reparse;
mod retirement;
mod selection;
mod small_fields;
mod superseded;
mod tool_blocks;

struct Fixture {
    dir: tempfile::TempDir,
    root: PathBuf,
    writer: Writer,
    head: parser_state::Head,
    origin: Origin,
    session: SessionId,
    message: MessageId,
}

/// Marks the prepared body ready and selected without running election.
const SELECT_PREPARED_BODY: &str = "BEGIN; UPDATE message_body SET ready=1,version_hash=(SELECT version_hash FROM message_candidate WHERE pk=message_body.preferred_candidate_pk); UPDATE message_candidate SET body_ready=1,ready=1,eligible=1; UPDATE message SET selected_body_pk=(SELECT pk FROM message_body); COMMIT;";

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let mut writer = Writer::open(&root).unwrap();
        let (head, _) = start_input(
            &mut writer,
            b"{\"synthetic\":true}\n",
            InputFormat::CodexRollout,
            &[],
        );
        let origin = Origin::new("synthetic").unwrap();
        let session = SessionId::new(&origin, "session:with:colons").unwrap();
        let message = MessageId::new(session.clone(), "message:with:colons").unwrap();
        Self {
            dir,
            root,
            writer,
            head,
            origin,
            session,
            message,
        }
    }
    /// Stores `raw` as another input and starts a parser on it.
    fn add_input(&mut self, raw: &[u8], format: InputFormat, context: &[&str]) -> String {
        let (head, id) = start_input(&mut self.writer, raw, format, context);
        self.head = head;
        id
    }
    /// Restarts the parser on the current input with a new version and digest.
    fn restart_parser(&mut self, parser_version: u32, digest: u8) {
        self.head = self
            .writer
            .begin_parser(&parser_state::Basis {
                input_pk: self.head.token.input_pk,
                chain: self.head.token.chain,
                parser_version,
                digest: [digest; 32],
                acquired_upto: self.head.acquired_upto,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
    }
    /// Commits one parser step that records `parsed_upto` bytes as parsed.
    fn advance_parser_to(&mut self, parsed_upto: u64) {
        let path = self.index_path();
        self.head = parser_state::write_step(
            &mut self.writer._index,
            &path,
            &self.head,
            parser_state::Progress {
                inspected_upto: parsed_upto,
                parsed_upto,
                phase: parser_state::Phase::Parsing,
                finalization_pending: true,
                counts: Default::default(),
            },
            &[],
            parser_state::Budget {
                cells: 0,
                encoded_bytes: 0,
            },
            |_| Ok(()),
        )
        .unwrap();
    }
    fn advance_parser(&mut self) {
        self.advance_parser_to(0);
    }
    fn prepare(&mut self) -> Result<Preparation> {
        self.prepare_with(|_, _| {})
    }
    fn prepare_with<'a>(
        &'a mut self,
        change: impl FnOnce(&mut Identity<'a>, &mut Source<'a>),
    ) -> Result<Preparation> {
        let mut identity = identity(&self.session, &self.origin, &self.message);
        let mut source = source(&self.head.token);
        change(&mut identity, &mut source);
        self.writer.prepare_message(identity, source, 4096)
    }
    fn reopen(self) -> Self {
        let Self {
            dir,
            root,
            writer,
            head,
            origin,
            session,
            message,
        } = self;
        drop(writer);
        let writer = crate::tests::reopen_after_drop(&root);
        Self {
            dir,
            root,
            writer,
            head,
            origin,
            session,
            message,
        }
    }
    fn index_path(&self) -> PathBuf {
        self.root.join("index.db")
    }
    fn scalar(&self, sql: &str) -> i64 {
        self.writer._index.query_row(sql, [], |r| r.get(0)).unwrap()
    }
    fn exec(&self, sql: &str) {
        self.writer._index.execute_batch(sql).unwrap();
    }
    fn select_prepared_body(&self) {
        self.exec(SELECT_PREPARED_BODY);
    }
    fn sequence(&self, build: &Build) -> u64 {
        self.writer
            .message_projection(build, 4096)
            .unwrap()
            .sequence
    }
    /// Writes, validates and seals one active value for `build`.
    fn seal(
        &mut self,
        build: &Build,
        purpose: ValuePurpose,
        codec: values::Codec,
        bytes: &[u8],
    ) -> values::Seal {
        let length = bytes.len() as u64;
        let head = match purpose {
            ValuePurpose::Normalized => self.writer.begin_message_value(build, codec, length),
            ValuePurpose::Origin => self.writer.begin_message_origin_value(build, length),
        }
        .unwrap();
        for (i, chunk) in bytes.chunks(values::MAX_CHUNK).enumerate() {
            self.writer
                .append_value_chunk(
                    &head.id,
                    (i * values::MAX_CHUNK) as u64,
                    chunk,
                    values::MAX_CHUNK,
                )
                .unwrap();
        }
        self.seal_written(build, purpose, &head.id)
    }
    /// Validates and seals a value whose bytes are already appended.
    fn seal_written(
        &mut self,
        build: &Build,
        purpose: ValuePurpose,
        id: &values::ValueId,
    ) -> values::Seal {
        let reader = self.writer.read_handle().unwrap();
        let mut proof = reader.validate_value(id).unwrap();
        while !proof.progress().complete {
            proof.advance(&reader, values::MAX_CHUNK).unwrap();
        }
        drop(reader);
        match purpose {
            ValuePurpose::Normalized => self.writer.seal_message_value(build, &proof),
            ValuePurpose::Origin => self.writer.seal_message_origin_value(build, &proof),
        }
        .unwrap()
    }
    /// A UTF-8 normalized value.
    fn value(&mut self, build: &Build, text: &[u8]) -> values::Seal {
        self.seal(build, ValuePurpose::Normalized, values::Codec::Utf8, text)
    }
    /// A JSON value. The bytes are opaque to the store; no selector meaning.
    fn json(&mut self, build: &Build, purpose: ValuePurpose, bytes: &[u8]) -> values::ReadyValue {
        self.seal(build, purpose, values::Codec::Json, bytes).ready
    }
    fn finish_json(
        &mut self,
        build: &Build,
        purpose: ValuePurpose,
        id: &values::ValueId,
    ) -> values::ReadyValue {
        self.seal_written(build, purpose, id).ready
    }
    /// Finishes the projection at its current cursor with a fixed hash.
    fn finish(&mut self, build: &Build) -> PreparedProjection {
        let sequence = self.sequence(build);
        self.writer
            .finish_message_projection(
                build,
                &step(sequence, b"source done"),
                polylogue_model::BlobHash::of(b"synthetic evidence unchanged"),
                polylogue_model::merge::FinishState::Unknown,
                4096,
            )
            .unwrap()
    }
    fn retire_until_complete(&mut self, build: &Build) {
        for _ in 0..1000 {
            if self.writer.retire_message_turn(build).unwrap() == Retirement::Complete {
                return;
            }
        }
        panic!("retirement did not complete");
    }
    fn collect_nodes(&mut self) {
        for _ in 0..1000 {
            if !self.writer.collect_message_nodes(&self.message).unwrap() {
                return;
            }
        }
        panic!("node collection did not finish");
    }
    fn collect_values(&mut self) {
        for _ in 0..1000 {
            if self.writer.collect_retiring_value().unwrap() == ValueCollection::Idle {
                return;
            }
        }
        panic!("value collection did not finish");
    }
}

fn start_input(
    writer: &mut Writer,
    raw: &[u8],
    format: InputFormat,
    context: &[&str],
) -> (parser_state::Head, String) {
    let blob = writer.put_file_blob(raw).unwrap();
    let input = writer
        .store_immutable_input(
            blob,
            format,
            context,
            &ImportObservation {
                id: &String::from_utf8_lossy(raw),
                machine: "synthetic",
                host: "synthetic",
                path: "synthetic/input.jsonl",
                seen_at: 0,
            },
            4096,
        )
        .unwrap();
    let head = writer
        .begin_parser(&parser_state::Basis {
            input_pk: input.input_pk,
            chain: input.chain,
            parser_version: 3,
            digest: [7; 32],
            acquired_upto: input.length,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    (head, input.id)
}

fn identity<'a>(
    session: &'a SessionId,
    origin: &'a Origin,
    message: &'a MessageId,
) -> Identity<'a> {
    Identity {
        session,
        origin,
        native_session: "session:with:colons",
        native_session_encoding: None,
        session_kind: SessionKind::Main,
        message,
        sequence: 0,
        role: Role::Assistant,
        written_by: WrittenBy::Model,
    }
}

fn source(token: &parser_state::Token) -> Source<'_> {
    Source {
        token,
        revision: 1,
        evidence_revision: 1,
        logical_key: [3; 32],
        route: RouteKind::Native,
    }
}

fn step(expected_sequence: u64, next_cursor: &[u8]) -> ProjectionStep<'_> {
    ProjectionStep {
        expected_sequence,
        next_cursor,
    }
}

fn presence(source_block: bool, source_text: bool) -> BlockPresence {
    BlockPresence {
        source_block,
        source_text,
    }
}
