//! The predecessor's `user.db`. Its assertions become assertions here with
//! their identities, authors, times and revision history; every row of
//! every table is also kept verbatim as one imported input. A row that
//! cannot be expressed here is refused with the reason, never changed to
//! fit, and stays readable in the kept rows.
//!
//! How the predecessor's rows map:
//! - a `judgment` whose target is another assertion and whose value carries
//!   a `decision` is a judgment of that exact revision;
//! - a row that supersedes another row about the same target is a new
//!   revision of it (the predecessor's promotion of a detector candidate);
//! - any other `judgment` is a request for one: a `question`;
//! - `blocker` is a `problem`; `finding` and `note` keep their kind;
//! - every stored column goes into the value under `predecessor`.
//!
//! Authors map to the nearest honest author: a detector or analysis run to
//! `script:<its stored name>`, a recorded agent session to that session,
//! and the operator's account name to `unattributed`, because commands run
//! by agents under the operator's account were stored under that name too.

use crate::older_archive::{json_value, open_source, tables_of};
use crate::{Result, StoreError};
use polylogue_model::assertion::values::{Claim, Judgment, Structured, WithPredecessor};
use polylogue_model::assertion::{AssertionFact, AssertionKind, AssertionLink, LinkRole};
use polylogue_model::authorship::Channel;
use polylogue_model::{Author, Json, Reference, RevisionReference, SessionId, Timestamp};
use rusqlite::Connection;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

/// The columns of the predecessor's `assertions` table.
pub const ASSERTION_COLUMNS: [&str; 18] = [
    "assertion_id",
    "scope_ref",
    "target_ref",
    "key",
    "kind",
    "value_json",
    "body_text",
    "author_ref",
    "author_kind",
    "evidence_refs_json",
    "status",
    "visibility",
    "confidence",
    "staleness_json",
    "context_policy_json",
    "supersedes_json",
    "created_at_ms",
    "updated_at_ms",
];

/// One predecessor assertion: the columns the mapping reads, and every
/// column as stored.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub id: String,
    pub target: String,
    pub kind: String,
    pub key: Option<String>,
    pub value: Option<String>,
    pub body: Option<String>,
    pub author: String,
    pub author_kind: String,
    pub evidence: Option<String>,
    pub staleness: Option<String>,
    pub supersedes: Option<String>,
    pub created_at: i64,
    pub stored: serde_json::Map<String, Json>,
}

/// One predecessor annotation schema.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub id: String,
    pub version: i64,
    pub definition: String,
    pub stored: Json,
}

/// The result of mapping one row: its form here, or why it has none.
pub type Mapped<T> = std::result::Result<T, String>;

pub struct UserDb {
    conn: Connection,
    _copies: Option<tempfile::TempDir>,
}

impl UserDb {
    /// Opens the predecessor's `user.db` at `path`. `scratch` is where a
    /// database with a write-ahead log holding frames is copied.
    pub fn open(path: &Path, scratch: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(StoreError::Invalid(format!(
                "{} is not a file: name the predecessor's user.db",
                path.display()
            )));
        }
        let mut copies = None;
        let conn = open_source(path, &mut copies, scratch)?;
        let declared = tables_of(&conn)?.get("assertions").is_some_and(|columns| {
            ASSERTION_COLUMNS
                .iter()
                .all(|c| columns.iter().any(|n| n == c))
        });
        if !declared {
            return Err(StoreError::Invalid(format!(
                "{} is not the predecessor's user.db: it has no assertions table with the predecessor's columns",
                path.display()
            )));
        }
        conn.execute_batch("BEGIN")?;
        Ok(Self {
            conn,
            _copies: copies,
        })
    }

    pub fn assertions(&self) -> Result<Vec<Row>> {
        let mut statement = self.conn.prepare(&format!(
            "SELECT {} FROM assertions ORDER BY created_at_ms, assertion_id",
            ASSERTION_COLUMNS.join(",")
        ))?;
        let mut rows = statement.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let mut stored = serde_json::Map::new();
            for (index, name) in ASSERTION_COLUMNS.iter().enumerate() {
                stored.insert((*name).to_owned(), json_value(row.get_ref(index)?));
            }
            out.push(Row {
                id: row.get(0)?,
                target: row.get(2)?,
                key: row.get(3)?,
                kind: row.get(4)?,
                value: row.get(5)?,
                body: row.get(6)?,
                author: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
                author_kind: row.get::<_, Option<String>>(8)?.unwrap_or_default(),
                evidence: row.get(9)?,
                staleness: row.get(13)?,
                supersedes: row.get(15)?,
                created_at: row.get(16)?,
                stored,
            });
        }
        Ok(out)
    }

    /// The member references of each stored result set, in rank order.
    pub fn result_set_members(&self) -> Result<BTreeMap<String, Vec<String>>> {
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if !tables_of(&self.conn)?.contains_key("result_set_members") {
            return Ok(out);
        }
        let mut statement = self.conn.prepare(
            "SELECT result_set_id, member_ref FROM result_set_members ORDER BY result_set_id, rank",
        )?;
        for row in statement.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))? {
            let (set, member): (String, String) = row?;
            out.entry(set).or_default().push(member);
        }
        Ok(out)
    }

    /// The annotation schemas, by ID and version, with every column as stored.
    pub fn schemas(&self) -> Result<Vec<Schema>> {
        if !tables_of(&self.conn)?.contains_key("annotation_schemas") {
            return Ok(Vec::new());
        }
        let mut statement = self.conn.prepare(
            "SELECT schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
             FROM annotation_schemas ORDER BY schema_id, schema_version",
        )?;
        let rows = statement.query_map([], |r| {
            Ok(Schema {
                id: r.get(0)?,
                version: r.get(1)?,
                definition: r.get(2)?,
                stored: serde_json::json!({
                    "schema_id": r.get::<_, String>(0)?,
                    "schema_version": r.get::<_, i64>(1)?,
                    "definition_json": r.get::<_, String>(2)?,
                    "definition_sha256": r.get::<_, String>(3)?,
                    "registered_at_ms": r.get::<_, i64>(4)?,
                }),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Rows per table.
    pub fn row_counts(&self) -> Result<BTreeMap<String, u64>> {
        let mut out = BTreeMap::new();
        for table in tables_of(&self.conn)?.into_keys() {
            let n: i64 =
                self.conn
                    .query_row(&format!("SELECT count(*) FROM \"{table}\""), [], |r| {
                        r.get(0)
                    })?;
            out.insert(table, n as u64);
        }
        Ok(out)
    }

    /// Writes every row of every table to `out`, one JSON object per line
    /// (`{"table": …, "row": {column: value}}`), in table and column order,
    /// with values kept exactly as the older archive's rows input keeps them.
    pub fn write_rows(&self, out: &mut dyn Write) -> Result<u64> {
        let mut written = 0;
        for (table, columns) in tables_of(&self.conn)? {
            let order = (1..=columns.len())
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let mut statement = self
                .conn
                .prepare(&format!("SELECT * FROM \"{table}\" ORDER BY {order}"))?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let mut values = serde_json::Map::new();
                for (index, name) in columns.iter().enumerate() {
                    values.insert(name.clone(), json_value(row.get_ref(index)?));
                }
                serde_json::to_writer(
                    &mut *out,
                    &serde_json::json!({"table": table, "row": values}),
                )?;
                out.write_all(b"\n")?;
                written += 1;
            }
        }
        Ok(written)
    }
}

/// What one row becomes, before its links are resolved.
#[derive(Debug, Clone, PartialEq)]
enum Form {
    /// A judgment of the named row, keyed `accepted`, `rejected` or `deferred`.
    Judgment {
        of: String,
        key: &'static str,
    },
    /// A new revision of the named row.
    Revision {
        of: String,
    },
    Kind(AssertionKind),
}

fn json_list(text: Option<&str>, column: &str) -> Mapped<Vec<String>> {
    match text {
        None => Ok(Vec::new()),
        Some(text) => serde_json::from_str(text)
            .map_err(|error| format!("{column} is not a list of references: {error}")),
    }
}

fn value_of(row: &Row) -> Mapped<Option<Json>> {
    row.value
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|error| format!("value_json is not JSON: {error}"))
}

fn form(row: &Row) -> Mapped<Form> {
    let supersedes = json_list(row.supersedes.as_deref(), "supersedes_json")?;
    if let Some(of) = row.target.strip_prefix("assertion:")
        && row.kind == "judgment"
        && let Some(decision) = value_of(row)?
            .as_ref()
            .and_then(|value| value.get("decision"))
            .and_then(Json::as_str)
            .map(str::to_owned)
    {
        let key = match decision.as_str() {
            "accept" => "accepted",
            "reject" => "rejected",
            "defer" => "deferred",
            other => return Err(format!("judgment decision {other} has no counterpart here")),
        };
        return Ok(Form::Judgment {
            of: of.to_owned(),
            key,
        });
    }
    match supersedes.as_slice() {
        [] => {}
        [base] => {
            let of = base.strip_prefix("assertion:").ok_or_else(|| {
                format!("it supersedes {base}, which is not an assertion reference")
            })?;
            return Ok(Form::Revision { of: of.to_owned() });
        }
        _ => return Err("it supersedes several assertions at once".into()),
    }
    Ok(Form::Kind(match row.kind.as_str() {
        "judgment" => AssertionKind::Question,
        "blocker" => AssertionKind::Problem,
        "finding" => AssertionKind::Finding,
        "note" => AssertionKind::Note,
        other => return Err(format!("assertion kind {other} has no counterpart here")),
    }))
}

/// A predecessor session ID (`<origin>:<native>`) as a session here.
fn session(text: &str) -> Option<Reference> {
    let (origin, native) = text.split_once(':')?;
    let origin = super::origin(origin)?;
    SessionId::parse(&format!("{origin}:{native}"))
        .ok()
        .map(Reference::Session)
}

/// A predecessor message ID (`<origin>:<session native>:<message>`) as a
/// message here.
fn message(text: &str) -> Option<Reference> {
    let (origin, rest) = text.split_once(':')?;
    let origin = super::origin(origin)?;
    Reference::parse(&format!("message:{origin}:{rest}")).ok()
}

fn author(row: &Row) -> Mapped<Author> {
    match row.author_kind.as_str() {
        "user" => Ok(Author::Unattributed),
        "detector" if !row.author.is_empty() => Ok(Author::Script(row.author.clone())),
        "agent" => Ok(row
            .author
            .strip_prefix("agent:")
            .and_then(session)
            .and_then(|reference| match reference {
                Reference::Session(id) => Some(Author::Agent(Some(id))),
                _ => None,
            })
            .unwrap_or(Author::Unattributed)),
        other => Err(format!("author kind {other} has no counterpart here")),
    }
}

/// The predecessor's assertions as assertion facts here, in an order in
/// which every row comes after the rows it names, each with its outcome.
/// `pass` is the exact revision of the pass every fact names as its origin;
/// `members` gives the member references of each stored result set.
pub fn facts(
    rows: &[Row],
    machine: &str,
    pass: &RevisionReference,
    members: &BTreeMap<String, Vec<String>>,
) -> Vec<(String, Mapped<AssertionFact>)> {
    let by_id: BTreeMap<&str, &Row> = rows.iter().map(|row| (row.id.as_str(), row)).collect();
    let forms: BTreeMap<&str, Mapped<Form>> = rows
        .iter()
        .map(|row| (row.id.as_str(), form(row)))
        .collect();
    let root = |id: &str| -> String {
        let mut current = id;
        let mut seen = BTreeSet::new();
        while let Some(Ok(Form::Revision { of })) = forms.get(current) {
            if !seen.insert(current) {
                break;
            }
            current = of;
        }
        current.to_owned()
    };
    let kind_of = |id: &str| -> Mapped<AssertionKind> {
        let mut current = id;
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(current) {
                return Err("its revisions form a cycle".into());
            }
            match forms.get(current) {
                Some(Ok(Form::Kind(kind))) => return Ok(*kind),
                Some(Ok(Form::Judgment { .. })) => return Ok(AssertionKind::Judgment),
                Some(Ok(Form::Revision { of })) => current = of,
                Some(Err(reason)) => return Err(format!("it revises a refused row: {reason}")),
                None => return Err(format!("it revises {current}, which user.db does not hold")),
            }
        }
    };
    let assertion =
        |id: &str| Reference::Assertion(RevisionReference::exact(root(id), id.to_owned()));
    let reference = |text: &str| -> Option<Reference> {
        if let Some(id) = text.strip_prefix("assertion:") {
            return by_id.contains_key(id).then(|| assertion(id));
        }
        if let Some(rest) = text.strip_prefix("session:") {
            return session(rest);
        }
        if let Some(rest) = text.strip_prefix("message:") {
            return message(rest);
        }
        session(text)
    };

    let map = |row: &Row| -> Mapped<AssertionFact> {
        let form = forms[row.id.as_str()].clone()?;
        let value = value_of(row)?;
        let kind = match &form {
            Form::Judgment { .. } => AssertionKind::Judgment,
            Form::Revision { of } => kind_of(of)?,
            Form::Kind(kind) => *kind,
        };
        let link = |role, reference| AssertionLink {
            role,
            reference,
            seen_hash: None,
            quote: None,
        };
        let mut links = Vec::new();
        match &form {
            Form::Judgment { of, .. } => {
                if !by_id.contains_key(of.as_str()) {
                    return Err(format!("it judges {of}, which user.db does not hold"));
                }
                links.push(link(LinkRole::About, assertion(of)));
            }
            _ => {
                if let Some(target) = reference(&row.target) {
                    links.push(link(LinkRole::About, target));
                } else {
                    // A finding about an analysis is about the rows of the
                    // result set it was drawn from.
                    let set = value
                        .as_ref()
                        .and_then(|value| value.get("result_set_ref"))
                        .and_then(Json::as_str)
                        .and_then(|text| text.strip_prefix("result-set:"));
                    let rows = set.and_then(|set| members.get(set)).ok_or_else(|| {
                        format!("its target {} has no counterpart here", row.target)
                    })?;
                    for member in rows {
                        let target = reference(member).ok_or_else(|| {
                            format!("result set member {member} has no counterpart here")
                        })?;
                        links.push(link(LinkRole::About, target));
                    }
                }
            }
        }
        if let Form::Revision { of } = &form {
            links.push(link(LinkRole::Revises, assertion(of)));
        }
        let mut evidence = BTreeSet::new();
        for text in json_list(row.evidence.as_deref(), "evidence_refs_json")? {
            if let Some(target) = reference(&text)
                && evidence.insert(target.to_string())
            {
                links.push(link(LinkRole::Evidence, target));
            }
        }
        links.push(link(LinkRole::Origin, Reference::Assertion(pass.clone())));

        let predecessor = Json::Object(row.stored.clone());
        let text = || {
            row.body
                .clone()
                .filter(|text| !text.is_empty())
                .ok_or_else(|| "it has no text".to_owned())
        };
        let value = match kind {
            AssertionKind::Judgment => serde_json::to_value(Structured {
                fields: Judgment {
                    reason: row.body.clone().filter(|text| !text.is_empty()),
                    blind: None,
                },
                predecessor: Some(predecessor),
            }),
            AssertionKind::Question | AssertionKind::Finding => serde_json::to_value(Structured {
                fields: Claim {
                    text: text()?,
                    check: None,
                    outcomes: None,
                },
                predecessor: Some(predecessor),
            }),
            _ => serde_json::to_value(WithPredecessor {
                text: text()?,
                predecessor,
            }),
        }
        .map_err(|error| error.to_string())?;
        let key = match (&form, kind) {
            (Form::Judgment { key, .. }, _) => Some((*key).to_owned()),
            (_, AssertionKind::Note) => None,
            _ => row.key.clone().filter(|key| !key.is_empty()),
        };
        let valid_until = row
            .staleness
            .as_deref()
            .and_then(|text| serde_json::from_str::<Json>(text).ok())
            .and_then(|staleness| {
                staleness
                    .get("expires_at")
                    .and_then(Json::as_str)
                    .map(str::to_owned)
            })
            .map(|text| {
                Timestamp::parse_rfc3339(&text)
                    .ok_or_else(|| format!("expiry {text} is not a time"))
            })
            .transpose()?;
        let fact = AssertionFact {
            id: row.id.clone(),
            root: root(&row.id),
            kind,
            key,
            value: Some(value),
            author: author(row)?,
            via: Channel::Import,
            machine: machine.to_owned(),
            created_at: Some(
                Timestamp::from_unix_ms(row.created_at)
                    .ok_or_else(|| format!("time {} is out of range", row.created_at))?,
            ),
            valid_until,
            links,
        };
        fact.canonicalized().map_err(|error| error.to_string())
    };

    // Rows naming other rows (as target, base or evidence) follow them.
    let names = |row: &Row| -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut add = |text: &str| {
            if let Some(id) = text.strip_prefix("assertion:")
                && id != row.id
                && by_id.contains_key(id)
            {
                out.insert(id.to_owned());
            }
        };
        add(&row.target);
        for list in [row.supersedes.as_deref(), row.evidence.as_deref()] {
            for text in json_list(list, "").unwrap_or_default() {
                add(&text);
            }
        }
        out
    };
    let mut order = Vec::with_capacity(rows.len());
    let mut placed = BTreeSet::new();
    let mut waiting: Vec<&Row> = rows.iter().collect();
    while !waiting.is_empty() {
        let (ready, rest): (Vec<&Row>, Vec<&Row>) = waiting
            .into_iter()
            .partition(|row| names(row).iter().all(|id| placed.contains(id)));
        if ready.is_empty() {
            // Rows naming each other in a cycle keep their stored order.
            order.extend(rest);
            break;
        }
        for row in ready {
            placed.insert(row.id.clone());
            order.push(row);
        }
        waiting = rest;
    }
    order
        .into_iter()
        .map(|row| (row.id.clone(), map(row)))
        .collect()
}

/// The `pass` imported assertion rows name as their origin, so one withdrawal
/// removes that group. Its population time is the newest row's; the same
/// source rows produce the same pass. Shared rubrics have separate memberships.
pub fn pass(id: &str, rows: &[Row], machine: &str) -> Mapped<AssertionFact> {
    let newest = rows.iter().map(|row| row.created_at).max();
    let fact = AssertionFact {
        id: id.to_owned(),
        root: id.to_owned(),
        kind: AssertionKind::Pass,
        key: None,
        value: Some(serde_json::json!({
            "rubric": null,
            "population": null,
            "counts": {"assertions": rows.len()},
        })),
        author: Author::Script("polylogue-import".into()),
        via: Channel::Import,
        machine: machine.to_owned(),
        created_at: newest
            .map(|at| {
                Timestamp::from_unix_ms(at).ok_or_else(|| format!("time {at} is out of range"))
            })
            .transpose()?,
        valid_until: None,
        links: Vec::new(),
    };
    fact.canonicalized().map_err(|error| error.to_string())
}

/// The unit a rubric's rows are drawn from, for each kind of target the
/// predecessor's schemas named; the first kind a schema names that has a
/// unit here is its unit, and the rest stay in its predecessor fields.
fn unit_query(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "session" => "sessions",
        "message" => "messages",
        "block" => "blocks",
        "delegation" => "links where kind:spawn",
        "assertion" => "assertions",
        _ => return None,
    })
}

/// A predecessor annotation schema as a `rubric`: its fields keep their
/// names, types, enums, ranges, requirement and description; its abstain,
/// confidence and rationale fields become the rubric's allowances; its
/// description is the instructions. An absent registration time stays unknown.
/// Physical store membership is recorded separately from this immutable fact.
pub fn rubric(schema: &Schema, machine: &str) -> Mapped<AssertionFact> {
    use polylogue_model::assertion::values::{Field, FieldType, Rubric};
    let definition: Json = serde_json::from_str(&schema.definition)
        .map_err(|error| format!("definition_json is not JSON: {error}"))?;
    let text = |key: &str| definition.get(key).and_then(Json::as_str);
    let kinds: Vec<&str> = definition
        .get("target_ref_kinds")
        .and_then(Json::as_array)
        .map(|kinds| kinds.iter().filter_map(Json::as_str).collect())
        .unwrap_or_default();
    let unit = kinds
        .iter()
        .find_map(|kind| unit_query(kind))
        .ok_or_else(|| format!("none of its target kinds {kinds:?} has a unit here"))?;
    let abstain_field = text("abstain_field");
    let mut fields = Vec::new();
    let (mut confidence, mut rationale) = (false, false);
    for field in definition
        .get("fields")
        .and_then(Json::as_array)
        .ok_or("it declares no fields")?
    {
        let name = field
            .get("name")
            .and_then(Json::as_str)
            .ok_or("a field has no name")?;
        if Some(name) == abstain_field {
            continue;
        }
        if name == "confidence" {
            confidence = true;
            continue;
        }
        if name == "rationale" {
            rationale = true;
            continue;
        }
        let value_type: FieldType =
            serde_json::from_value(field.get("value_type").cloned().unwrap_or(Json::Null))
                .map_err(|_| format!("field {name} has a type with no counterpart here"))?;
        let number = |key: &str| field.get(key).and_then(Json::as_number).cloned();
        fields.push(Field {
            name: name.to_owned(),
            value_type,
            enum_values: field
                .get("enum_values")
                .and_then(Json::as_array)
                .filter(|values| !values.is_empty())
                .cloned(),
            minimum: number("minimum"),
            maximum: number("maximum"),
            required: field.get("required").and_then(Json::as_bool),
            description: field
                .get("description")
                .and_then(Json::as_str)
                .map(str::to_owned),
        });
    }
    let instructions = match (text("title"), text("description")) {
        (Some(title), Some(description)) => format!("{title}. {description}"),
        (None, Some(description)) => description.to_owned(),
        (Some(title), None) => title.to_owned(),
        (None, None) => return Err("it has no description to instruct from".into()),
    };
    let value = serde_json::to_value(Structured {
        fields: Rubric {
            instructions,
            unit_query: unit.to_owned(),
            fields,
            abstain: abstain_field.is_some(),
            confidence,
            rationale: Some(rationale),
        },
        predecessor: Some(schema.stored.clone()),
    })
    .map_err(|error| error.to_string())?;
    let id = format!("predecessor-schema:{}@{}", schema.id, schema.version);
    let fact = AssertionFact {
        id: id.clone(),
        root: id,
        kind: AssertionKind::Rubric,
        key: Some(schema.id.clone()),
        value: Some(value),
        author: Author::Script("polylogue".into()),
        via: Channel::Import,
        machine: machine.to_owned(),
        created_at: match schema.stored.get("registered_at_ms").and_then(Json::as_i64) {
            None | Some(0) => None,
            Some(at) => Some(
                Timestamp::from_unix_ms(at).ok_or_else(|| format!("time {at} is out of range"))?,
            ),
        },
        valid_until: None,
        links: Vec::new(),
    };
    fact.canonicalized().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests;
