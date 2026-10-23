//! Stateless page continuation over one unchanged view and resolved question.
use super::{Failure, Options, Program};
use base64::Engine as _;
use polylogue_model::ChangePosition;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const VERSION: u8 = 1;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Cursor {
    version: u8,
    pub offset: u64,
    pub now: i64,
    pub seed: u64,
    position: ChangePosition,
    question: String,
}

fn invalid(message: &str) -> Failure {
    Failure {
        kind: "bad_request".into(),
        message: message.into(),
        detail: None,
    }
}

impl Cursor {
    pub fn new(
        offset: u64,
        now: i64,
        seed: u64,
        position: ChangePosition,
        question: String,
    ) -> Self {
        Self {
            version: VERSION,
            offset,
            now,
            seed,
            position,
            question,
        }
    }

    pub fn decode(text: &str) -> Result<Self, Failure> {
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(text)
            .map_err(|_| invalid("the page cursor is invalid"))?;
        let cursor: Self = serde_json::from_slice(&bytes)
            .map_err(|_| invalid("the page cursor is invalid or from an unsupported version"))?;
        if cursor.version != VERSION {
            return Err(invalid("the page cursor version is not supported"));
        }
        Ok(cursor)
    }

    pub fn encode(&self) -> Result<String, Failure> {
        let bytes = serde_json::to_vec(self)
            .map_err(|e| invalid(&format!("could not encode page cursor: {e}")))?;
        Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
    }

    pub fn check_view(&self, position: &ChangePosition) -> Result<(), Failure> {
        if &self.position != position {
            return Err(Failure {
                kind: "stale_cursor".into(),
                message: "the query view changed; run the query again".into(),
                detail: Some(serde_json::json!({"reason":"view_changed"})),
            });
        }
        Ok(())
    }

    pub fn check_question(&self, question: &str) -> Result<(), Failure> {
        if self.question != question {
            return Err(invalid(
                "the cursor belongs to a different query or query context",
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Question<'a> {
    program: Value,
    session: &'a Option<String>,
    fields: &'a [String],
    embeddings: bool,
    why: bool,
    #[serde(with = "jiff::fmt::serde::tz::required")]
    zone: &'a jiff::tz::TimeZone,
    day_start_minutes: u16,
}

pub(super) fn question(program: &Program, options: &Options, why: bool) -> Result<String, Failure> {
    let failed = |e: serde_json::Error| Failure {
        kind: "not_built".into(),
        message: format!("the query context cannot be encoded for page continuation: {e}"),
        detail: None,
    };
    let mut compiled = serde_json::to_value(program).map_err(failed)?;
    // Presentation may change between pages; selection remains in Program.sql.
    if let Some(fields) = compiled.as_object_mut() {
        fields.remove("page_size");
    }
    let binding = Question {
        program: compiled,
        session: &options.self_session,
        fields: &options.fields,
        embeddings: options.embeddings,
        why,
        zone: &options.clock.day.zone,
        day_start_minutes: options.clock.day.start_minutes,
    };
    let mut digest = blake3::Hasher::new();
    serde_json::to_writer(&mut digest, &binding).map_err(failed)?;
    Ok(digest.finalize().to_hex().to_string())
}

pub(super) fn semantic_paging() -> Failure {
    Failure { kind: "not_built".into(), message: "semantic ranking pagination is not available; request a page large enough for the selected answer".into(), detail: None }
}
