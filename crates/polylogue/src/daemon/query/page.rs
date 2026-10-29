//! Page continuation by key: a cursor holds the last row's order-key
//! values, and the next page reads the rows after them. Writes between
//! pages neither refuse the cursor nor repeat or skip rows whose keys did
//! not change.
use super::{Failure, Options, Program};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Cursor {
    /// The previous page's last row, one value per order key.
    pub after: Vec<Value>,
    pub now: i64,
    pub seed: u64,
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
    pub fn new(after: Vec<Value>, now: i64, seed: u64, question: String) -> Self {
        Self {
            after,
            now,
            seed,
            question,
        }
    }

    /// The cursor for the page after `row`, from its order-key columns.
    pub fn after_row(program: &Program, row: &Value, seed: u64, question: String) -> Self {
        let after = program
            .order
            .iter()
            .map(|key| row[key.column.as_str()].clone())
            .collect();
        Self::new(after, program.now, seed, question)
    }

    pub fn decode(text: &str) -> Result<Self, Failure> {
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(text)
            .map_err(|_| invalid("the page cursor is invalid"))?;
        serde_json::from_slice(&bytes).map_err(|_| invalid("the page cursor is invalid"))
    }

    pub fn encode(&self) -> Result<String, Failure> {
        let bytes = serde_json::to_vec(self)
            .map_err(|e| invalid(&format!("could not encode page cursor: {e}")))?;
        Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
    }

    /// Checks the cursor belongs to this query and returns its key values.
    pub fn check(&self, program: &Program, question: &str) -> Result<&[Value], Failure> {
        if self.question != question || self.after.len() != program.order.len() {
            return Err(invalid(
                "the cursor belongs to a different query or query context",
            ));
        }
        Ok(&self.after)
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
    /// For `near:`: the vectors the similarities came from. A page after
    /// vectors were stored or removed would rank by different values.
    vector_writes: Option<u64>,
}

pub(super) fn question(program: &Program, options: &Options, why: bool) -> Result<String, Failure> {
    let failed = |e: serde_json::Error| Failure {
        kind: "not_built".into(),
        message: format!("the query context cannot be encoded for page continuation: {e}"),
        detail: None,
    };
    let mut compiled = serde_json::to_value(program).map_err(failed)?;
    // Page size and count execution plans do not change the page's
    // selection or order, which remain in Program.sql.
    if let Some(fields) = compiled.as_object_mut() {
        for field in ["page_size", "row_count_sql", "row_membership_sql"] {
            fields.remove(field);
        }
    }
    let binding = Question {
        program: compiled,
        session: &options.self_session,
        fields: &options.fields,
        embeddings: options.embeddings,
        why,
        zone: &options.clock.day.zone,
        day_start_minutes: options.clock.day.start_minutes,
        vector_writes: program
            .near
            .is_some()
            .then(polylogue_store::vectors::writes),
    };
    let mut digest = blake3::Hasher::new();
    serde_json::to_writer(&mut digest, &binding).map_err(failed)?;
    Ok(digest.finalize().to_hex().to_string())
}
