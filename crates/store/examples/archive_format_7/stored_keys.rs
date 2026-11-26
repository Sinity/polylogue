//! Format 10: keys inside stored payloads that are also the wire's say what
//! they hold. Each payload is decoded as JSON, its keys are renamed at their
//! exact paths (`what`, `question`, `scope`, `generation` and `pos` mean
//! other things elsewhere, so no key is renamed by name alone), the result is
//! decoded with the build's types and written back, and the hashes that
//! cover a rewritten payload are computed again in place.
//!
//! | Payload | Earlier key | Key |
//! |---|---|---|
//! | saved query definitions, frozen definitions, run records | `question` | `query_id` |
//! | execution bindings | `scope_fingerprint`, `versions.scope` | `access_fingerprint`, `versions.access` |
//! | read positions, `ChangePosition` | `generation` | `index_instance` |
//! | Save, saved rows, run records | `read_at` | `read_position` |
//! | an effect's and a Save report's saved rows | `kept_results`, `kept` | `saved_rows` |
//! | delivery events, the delivery-event write | `what` | `outcome` |
//! | what a delivery left out (`left_out_json` and every copy) | `what` | `item` |
//! | signal events | `ordinal` | `request_position` |
//! | slot annotations | `pos` | `position` |
//!
//! Recomputed in place: `client_request.content_hash`, and
//! `query_run.binding_hash` and `run_hash` with their copies in the
//! completion. No identifier another row cites changes: a query's ID is a
//! hash of the shared query language's normal form, which holds none of
//! these keys, and the step proves it for every stored query object and its
//! definitions before it rewrites anything. Request, run, delivery and
//! annotation IDs are chosen when written, and the signal checkpoints whose
//! hashes chain through `Commit.prior` hold no renamed key.
//!
//! Not renamed, on purpose: the keys of `query_language::cursor::CursorContext`
//! (its `question`, `read_positions` and `versions` names; only Polylogue's
//! own entries inside the two maps change), which is the shared crate's type;
//! the query language's unit columns (`slots.pos`, `runs.read_at`), which are
//! words users and saved queries write; the client's own request envelope
//! kept with a line or a binding (`resolve_line.request`,
//! `accepted_request`), which records what the caller sent.
use polylogue_model::authored::AuthoredEffect;
use polylogue_model::query_objects::{QueryObject, QueryRun, QueryRunOutcome};
use polylogue_model::signal::EventRecord;
use polylogue_model::user::{LeftOut, UserRequest};
use query_language::identity::{QuestionId, question_id_from_normal_form};
use rusqlite::types::Value as Column;
use rusqlite::{Connection, params};
use serde_json::Value;
use std::collections::BTreeMap;

type Error = Box<dyn std::error::Error>;

/// The saved rows table and its read position column, by their format-9
/// names.
const SAVED_ROWS: (&str, &str) = ("kept_result", "read_at");

/// Why the step stops before or after rewriting.
#[derive(Debug)]
pub(super) enum Refusal {
    /// A stored query object's ID is not the hash of its stored normal form.
    QueryId { query: String, recomputed: String },
    /// A definition inside a query object names an ID its body does not hash to.
    DefinitionId {
        query: String,
        declared: String,
        recomputed: String,
    },
    /// Rows keep an earlier key after the step.
    EarlierShape { table: &'static str, rows: u64 },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueryId { query, recomputed } => write!(
                f,
                "query {query} does not hash to its ID from its stored normal form ({recomputed})"
            ),
            Self::DefinitionId {
                query,
                declared,
                recomputed,
            } => write!(
                f,
                "query {query} holds a definition declared {declared} that hashes to {recomputed}"
            ),
            Self::EarlierShape { table, rows } => {
                write!(f, "{rows} rows of {table} keep an earlier key")
            }
        }
    }
}

impl std::error::Error for Refusal {}

/// Renames `old` to `new` in an object, keeping its place among the keys;
/// one when the key was there.
fn rename(value: &mut Value, old: &str, new: &str) -> u64 {
    let Value::Object(map) = value else { return 0 };
    if !map.contains_key(old) {
        return 0;
    }
    let entries = std::mem::take(map);
    *map = entries
        .into_iter()
        .map(|(key, value)| (if key == old { new.to_owned() } else { key }, value))
        .collect();
    1
}

fn items(value: Option<&mut Value>) -> impl Iterator<Item = &mut Value> {
    value
        .and_then(Value::as_array_mut)
        .into_iter()
        .flat_map(|items| items.iter_mut())
}

/// A `ChangePosition`.
fn position(value: Option<&mut Value>) -> u64 {
    value.map_or(0, |value| rename(value, "generation", "index_instance"))
}

/// The two maps of a replay context hold Polylogue's own entries.
fn context(value: Option<&mut Value>) -> u64 {
    let Some(context) = value else { return 0 };
    let mut n = 0;
    if let Some(positions) = context.get_mut("read_positions") {
        n += rename(positions, "generation", "index_instance");
    }
    if let Some(versions) = context.get_mut("versions") {
        n += rename(versions, "scope", "access");
    }
    n
}

/// An `ExecutionBinding`.
fn execution(value: Option<&mut Value>) -> u64 {
    let Some(replay) = value.and_then(|value| value.get_mut("replay")) else {
        return 0;
    };
    let mut n = rename(replay, "scope_fingerprint", "access_fingerprint");
    for definition in items(replay.get_mut("definitions")) {
        n += rename(definition, "question", "query_id");
    }
    n + context(replay.get_mut("context"))
}

/// An `ObjectExecutionBinding`.
fn object_execution(value: Option<&mut Value>) -> u64 {
    let Some(execution) = value else { return 0 };
    let mut n = rename(execution, "scope_fingerprint", "access_fingerprint");
    if let Some(versions) = execution.get_mut("versions") {
        n += rename(versions, "scope", "access");
    }
    n
}

/// A `QueryRun`.
pub(super) fn run(value: &mut Value) -> u64 {
    let mut n = rename(value, "question", "query_id") + rename(value, "read_at", "read_position");
    n += position(value.get_mut("read_position"));
    if let Some(binding) = value.get_mut("binding") {
        let kind = binding
            .get("kind")
            .and_then(Value::as_str)
            .map(str::to_owned);
        n += match kind.as_deref() {
            Some("original") => execution(binding.get_mut("execution")),
            Some("original_object") => object_execution(binding.get_mut("execution")),
            _ => 0,
        };
    }
    n
}

/// A `QueryObject`.
fn query_object(value: &mut Value) -> u64 {
    items(value.get_mut("definitions"))
        .map(|definition| rename(definition, "question", "query_id"))
        .sum()
}

/// A list of `LeftOut`.
fn left_out(value: Option<&mut Value>) -> u64 {
    items(value)
        .map(|left_out| rename(left_out, "what", "item"))
        .sum()
}

/// A slot annotation's value.
fn slot(value: Option<&mut Value>) -> u64 {
    value.map_or(0, |value| rename(value, "pos", "position"))
}

/// JSON kept as text inside a payload, rewritten by `change` when it renames
/// something.
fn inside_text(value: Option<&mut Value>, change: impl FnOnce(&mut Value) -> u64) -> u64 {
    let Some(Value::String(text)) = value else {
        return 0;
    };
    let Ok(mut inner) = serde_json::from_str::<Value>(text) else {
        return 0;
    };
    let n = change(&mut inner);
    if n > 0 {
        *text = inner.to_string();
    }
    n
}

/// A saved body (`rows_json`): its read position, as the listing reply has it.
fn saved_body(value: &mut Value) -> u64 {
    rename(value, "read_at", "read_position") + position(value.get_mut("read_position"))
}

/// A Save's read position text: the clock and the listing's position.
fn saved_read_position(value: &mut Value) -> u64 {
    position(value.get_mut("position"))
}

/// The two texts a Save and a saved row keep.
fn saved_texts(value: &mut Value) -> u64 {
    let mut n = rename(value, "read_at", "read_position");
    n += inside_text(value.get_mut("rows_json"), saved_body);
    n + inside_text(value.get_mut("read_position"), saved_read_position)
}

fn new_annotation(value: &mut Value) -> u64 {
    if value.get("kind").and_then(Value::as_str) == Some("slot") {
        slot(value.get_mut("value"))
    } else {
        0
    }
}

/// A `UserRequest`. A revision's patch names no kind; `slots` says whether
/// the request revised slots, as its effect records.
pub(super) fn request(value: &mut Value, slots: bool) -> u64 {
    let Some(write) = value.get_mut("write") else {
        return 0;
    };
    let tag = write
        .get("write")
        .and_then(Value::as_str)
        .map(str::to_owned);
    match tag.as_deref() {
        Some("save") => {
            let mut n = saved_texts(write);
            if let Some(outcomes) = write.get_mut("outcomes") {
                n += rename(outcomes, "kept", "saved_rows");
            }
            n
        }
        Some("mark_seen") => items(write.get_mut("marks"))
            .map(|mark| position(mark.get_mut("at")))
            .sum(),
        Some("resolve_line") => execution(
            write
                .get_mut("retained")
                .and_then(|retained| retained.get_mut("binding")),
        ),
        Some("delivery_event") => rename(write, "what", "outcome"),
        Some("deliver" | "delivered_advice") => left_out(write.get_mut("left_out")),
        Some("signal_commit") => items(write.get_mut("pending_delivery"))
            .map(|pending| {
                left_out(
                    pending
                        .get_mut("record")
                        .and_then(|r| r.get_mut("left_out")),
                )
            })
            .sum(),
        Some("assert") => items(write.get_mut("annotations"))
            .map(new_annotation)
            .sum(),
        Some("assert_query") => write.get_mut("annotation").map_or(0, new_annotation),
        Some("revise_annotations") if slots => {
            let patch = write
                .get_mut("patch")
                .and_then(|patch| patch.get_mut("value"));
            match patch {
                Some(change) if change.get("change").and_then(Value::as_str) == Some("set") => {
                    slot(change.get_mut("value"))
                }
                _ => 0,
            }
        }
        _ => 0,
    }
}

/// An `AuthoredEffect`.
pub(super) fn effect(value: &mut Value) -> u64 {
    let mut n = items(value.get_mut("annotations"))
        .map(new_annotation)
        .sum::<u64>();
    for event in items(value.get_mut("signal_events")) {
        n += rename(event, "ordinal", "request_position");
    }
    n += rename(value, "kept_results", "saved_rows");
    for saved in items(value.get_mut("saved_rows")) {
        n += saved_texts(saved);
    }
    for change in items(value.get_mut("deliveries")) {
        if let Some(delivery) = change.get_mut("delivery") {
            for event in items(delivery.get_mut("events")) {
                n += rename(event, "what", "outcome");
            }
            n += left_out(
                delivery
                    .get_mut("prepared")
                    .and_then(|prepared| prepared.get_mut("left_out")),
            );
        }
        for event in items(change.get_mut("added_events")) {
            n += rename(event, "what", "outcome");
        }
    }
    n
}

/// Whether an effect revised slot annotations.
fn revises_slots(effect: &Value) -> bool {
    effect["annotations"]
        .as_array()
        .is_some_and(|all| all.iter().any(|a| a["kind"] == "slot"))
}

fn unpack(bytes: &[u8]) -> Result<Value, Error> {
    Ok(serde_json::from_slice(&zstd::decode_all(bytes)?)?)
}

fn pack(value: &impl serde::Serialize) -> Result<Vec<u8>, Error> {
    Ok(zstd::bulk::compress(&serde_json::to_vec(value)?, 3)?)
}

/// Rows of `table` a page at a time in key order, each page read whole
/// before `visit` sees its rows; `visit` may update the rows it is given.
fn each_row(
    conn: &Connection,
    table: &str,
    key: &str,
    columns: &[&str],
    filter: &str,
    mut visit: impl FnMut(&str, &[Column]) -> Result<(), Error>,
) -> Result<u64, Error> {
    const PAGE: i64 = 512;
    let sql = format!(
        "SELECT {key}, {} FROM main.{table} WHERE {key} > ?1 {filter} ORDER BY {key} LIMIT ?2",
        columns.join(", ")
    );
    let mut after = String::new();
    let mut visited = 0;
    loop {
        let page: Vec<(String, Vec<Column>)> = conn
            .prepare_cached(&sql)?
            .query_map(params![after, PAGE], |row| {
                let values = (1..=columns.len())
                    .map(|i| row.get::<_, Column>(i))
                    .collect::<rusqlite::Result<_>>()?;
                Ok((row.get(0)?, values))
            })?
            .collect::<rusqlite::Result<_>>()?;
        let Some((last, _)) = page.last() else {
            return Ok(visited);
        };
        after = last.clone();
        for (id, values) in page {
            visited += 1;
            visit(&id, &values)?;
        }
    }
}

fn blob(column: &Column) -> Option<&[u8]> {
    match column {
        Column::Blob(bytes) => Some(bytes),
        _ => None,
    }
}

fn text(column: &Column) -> Option<&str> {
    match column {
        Column::Text(text) => Some(text),
        _ => None,
    }
}

/// One table's work: what it visited and rewrote, and its rows in the
/// earlier shape before and after.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub visited: u64,
    pub rewritten: u64,
    pub before: u64,
    pub after: u64,
}

/// A table the step rewrites: how to count its rows in the earlier shape and
/// how to rewrite them.
struct Table {
    name: &'static str,
    earlier: fn(&Connection) -> Result<u64, Error>,
    rewrite: fn(&Connection) -> Result<(u64, u64), Error>,
}

const TABLES: &[Table] = &[
    Table {
        name: "client_request",
        earlier: requests_earlier,
        rewrite: requests,
    },
    Table {
        name: "query",
        earlier: queries_earlier,
        rewrite: queries,
    },
    Table {
        name: "query_run",
        earlier: runs_earlier,
        rewrite: runs,
    },
    Table {
        name: "saved rows",
        earlier: saved_rows_earlier,
        rewrite: saved_rows,
    },
    Table {
        name: "delivery",
        earlier: deliveries_earlier,
        rewrite: deliveries,
    },
    Table {
        name: "annotation",
        earlier: slots_earlier,
        rewrite: slots,
    },
    Table {
        name: "signal_event",
        earlier: signal_events_earlier,
        rewrite: signal_events,
    },
];

/// The step: proves the query IDs, rewrites each table, and refuses unless
/// no row keeps an earlier key.
pub(super) fn step(conn: &Connection, _: &super::Context<'_>) -> Result<(), Error> {
    let proven = prove_query_ids(conn)?;
    println!("stored keys\tquery objects whose IDs hash from their normal forms\t{proven}\t-");
    for (table, counts) in convert(conn)? {
        println!("stored keys\t{table} rows visited\t{}\t-", counts.visited);
        println!(
            "stored keys\t{table} rows rewritten\t-\t{}",
            counts.rewritten
        );
        println!(
            "stored keys\t{table} rows in the earlier shape\t{}\t{}",
            counts.before, counts.after
        );
    }
    Ok(())
}

/// Rewrites every table; refuses when a row keeps an earlier key after.
pub(super) fn convert(conn: &Connection) -> Result<Vec<(&'static str, Counts)>, Error> {
    let mut all = Vec::new();
    for table in TABLES {
        let before = (table.earlier)(conn)?;
        let (visited, rewritten) = (table.rewrite)(conn)?;
        let after = (table.earlier)(conn)?;
        if after != 0 {
            return Err(Refusal::EarlierShape {
                table: table.name,
                rows: after,
            }
            .into());
        }
        all.push((
            table.name,
            Counts {
                visited,
                rewritten,
                before,
                after,
            },
        ));
    }
    Ok(all)
}

fn hex(id: &QuestionId) -> String {
    id.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Every stored query object's ID, and each of its definitions', is the hash
/// of its stored normal form and parameters: the step renames no key a
/// query ID is computed over. Read before anything is rewritten.
pub(super) fn prove_query_ids(conn: &Connection) -> Result<u64, Error> {
    let parameters = |value: &Value| -> Result<BTreeMap<String, String>, Error> {
        Ok(serde_json::from_value(value.clone())?)
    };
    each_row(
        conn,
        "query",
        "hash",
        &["normal_form", "object"],
        "",
        |hash, row| {
            let normal_form: Value = serde_json::from_str(text(&row[0]).ok_or("normal form")?)?;
            let object = unpack(blob(&row[1]).ok_or("query object")?)?;
            let recomputed = hex(&question_id_from_normal_form(
                &normal_form,
                &parameters(&object["parameters"])?,
            )?);
            let declared: QuestionId = serde_json::from_value(object["id"].clone())?;
            if recomputed != hash || hex(&declared) != hash {
                return Err(Refusal::QueryId {
                    query: hash.to_owned(),
                    recomputed,
                }
                .into());
            }
            for definition in object["definitions"].as_array().into_iter().flatten() {
                let declared = definition
                    .get("question")
                    .or_else(|| definition.get("query_id"))
                    .ok_or("a definition without its ID")?;
                let declared = hex(&serde_json::from_value(declared.clone())?);
                let recomputed = hex(&question_id_from_normal_form(
                    &definition["normal_form"],
                    &parameters(&definition["parameters"])?,
                )?);
                if declared != recomputed {
                    return Err(Refusal::DefinitionId {
                        query: hash.to_owned(),
                        declared,
                        recomputed,
                    }
                    .into());
                }
            }
            Ok(())
        },
    )
}

fn requests_earlier(conn: &Connection) -> Result<u64, Error> {
    let mut n = 0;
    each_row(
        conn,
        "client_request",
        "id",
        &["request", "effect"],
        "",
        |_, row| {
            let mut effect_value = blob(&row[1]).map(unpack).transpose()?;
            let slots = effect_value.as_ref().is_some_and(revises_slots);
            let mut renamed = blob(&row[0]).map_or(Ok(0), |bytes| {
                unpack(bytes).map(|mut value| request(&mut value, slots))
            })?;
            renamed += effect_value.as_mut().map_or(0, effect);
            n += u64::from(renamed > 0);
            Ok(())
        },
    )?;
    Ok(n)
}

/// Requests and their effects, with the request's content hash again.
fn requests(conn: &Connection) -> Result<(u64, u64), Error> {
    let mut rewritten = 0;
    let visited = each_row(
        conn,
        "client_request",
        "id",
        &["request", "effect"],
        "",
        |id, row| {
            let Some(bytes) = blob(&row[0]) else {
                return Ok(());
            };
            let mut request_value = unpack(bytes)?;
            let mut effect_value = blob(&row[1]).map(unpack).transpose()?;
            let slots = effect_value.as_ref().is_some_and(revises_slots);
            let renamed =
                request(&mut request_value, slots) + effect_value.as_mut().map_or(0, effect);
            if renamed == 0 {
                return Ok(());
            }
            let typed: UserRequest = serde_json::from_value(request_value)?;
            let hash = polylogue_store::authored_records::request_hash(&typed)?;
            let effect_bytes = effect_value
                .map(|value| -> Result<_, Error> {
                    let typed: AuthoredEffect = serde_json::from_value(value)?;
                    pack(&typed)
                })
                .transpose()?;
            conn.execute(
            "UPDATE main.client_request SET request = ?2, effect = coalesce(?3, effect), content_hash = ?4 WHERE id = ?1",
            params![id, pack(&typed)?, effect_bytes, &hash[..]],
        )?;
            rewritten += 1;
            Ok(())
        },
    )?;
    Ok((visited, rewritten))
}

fn queries_earlier(conn: &Connection) -> Result<u64, Error> {
    let mut n = 0;
    each_row(conn, "query", "hash", &["object"], "", |_, row| {
        let mut value = unpack(blob(&row[0]).ok_or("query object")?)?;
        n += u64::from(query_object(&mut value) > 0);
        Ok(())
    })?;
    Ok(n)
}

fn queries(conn: &Connection) -> Result<(u64, u64), Error> {
    let mut rewritten = 0;
    let visited = each_row(conn, "query", "hash", &["object"], "", |hash, row| {
        let mut value = unpack(blob(&row[0]).ok_or("query object")?)?;
        if query_object(&mut value) == 0 {
            return Ok(());
        }
        let typed: QueryObject = serde_json::from_value(value)?;
        conn.execute(
            "UPDATE main.query SET object = ?2 WHERE hash = ?1",
            params![hash, pack(&typed)?],
        )?;
        rewritten += 1;
        Ok(())
    })?;
    Ok((visited, rewritten))
}

fn runs_earlier(conn: &Connection) -> Result<u64, Error> {
    let mut n = 0;
    each_row(conn, "query_run", "id", &["run"], "", |_, row| {
        if let Some(bytes) = blob(&row[0]) {
            n += u64::from(run(&mut unpack(bytes)?) > 0);
        }
        Ok(())
    })?;
    Ok(n)
}

/// Runs, with their binding and record hashes and their completion's
/// copies computed again over the run as it now reads. A run an earlier
/// step rewrote has its hashes computed here too.
fn runs(conn: &Connection) -> Result<(u64, u64), Error> {
    let mut rewritten = 0;
    let visited = each_row(
        conn,
        "query_run",
        "id",
        &["run", "completion", "binding_hash", "run_hash"],
        "",
        |id, row| {
            let Some(bytes) = blob(&row[0]) else {
                return Ok(());
            };
            let mut value = unpack(bytes)?;
            let renamed = run(&mut value);
            let typed: QueryRun = serde_json::from_value(value)?;
            let (binding_hash, run_hash) = (
                typed.binding_hash()?.to_string(),
                typed.record_hash()?.to_string(),
            );
            if renamed == 0
                && text(&row[2]) == Some(binding_hash.as_str())
                && text(&row[3]) == Some(run_hash.as_str())
            {
                return Ok(());
            }
            conn.execute(
                "UPDATE main.query_run SET run = ?2, binding_hash = ?3, run_hash = ?4 WHERE id = ?1",
                params![id, pack(&typed)?, binding_hash, run_hash],
            )?;
            if let Some(completion) = blob(&row[1]) {
                let mut outcome: QueryRunOutcome = serde_json::from_value(unpack(completion)?)?;
                outcome.binding_hash = typed.binding_hash()?;
                outcome.run_hash = typed.record_hash()?;
                conn.execute(
                    "UPDATE main.query_run SET completion = ?2 WHERE id = ?1",
                    params![id, pack(&outcome)?],
                )?;
            }
            rewritten += 1;
            Ok(())
        },
    )?;
    Ok((visited, rewritten))
}

/// A saved row's two texts, as a payload with the two keys.
fn saved_row_value(row: &[Column]) -> Value {
    serde_json::json!({
        "rows_json": text(&row[0]),
        "read_position": text(&row[1]),
    })
}

fn saved_rows_earlier(conn: &Connection) -> Result<u64, Error> {
    let (table, column) = SAVED_ROWS;
    let mut n = 0;
    each_row(conn, table, "id", &["rows_json", column], "", |_, row| {
        n += u64::from(saved_texts(&mut saved_row_value(row)) > 0);
        Ok(())
    })?;
    Ok(n)
}

fn saved_rows(conn: &Connection) -> Result<(u64, u64), Error> {
    let (table, column) = SAVED_ROWS;
    let mut rewritten = 0;
    let visited = each_row(conn, table, "id", &["rows_json", column], "", |id, row| {
        let mut value = saved_row_value(row);
        if saved_texts(&mut value) == 0 {
            return Ok(());
        }
        conn.execute(
            &format!("UPDATE main.{table} SET rows_json = ?2, {column} = ?3 WHERE id = ?1"),
            params![
                id,
                value["rows_json"].as_str(),
                value["read_position"].as_str()
            ],
        )?;
        rewritten += 1;
        Ok(())
    })?;
    Ok((visited, rewritten))
}

fn left_out_value(row: &[Column]) -> Result<Value, Error> {
    Ok(serde_json::from_str(text(&row[0]).unwrap_or("[]"))?)
}

fn deliveries_earlier(conn: &Connection) -> Result<u64, Error> {
    let mut n = 0;
    each_row(conn, "delivery", "id", &["left_out_json"], "", |_, row| {
        n += u64::from(left_out(Some(&mut left_out_value(row)?)) > 0);
        Ok(())
    })?;
    Ok(n)
}

fn deliveries(conn: &Connection) -> Result<(u64, u64), Error> {
    let mut rewritten = 0;
    let visited = each_row(conn, "delivery", "id", &["left_out_json"], "", |id, row| {
        let mut value = left_out_value(row)?;
        if left_out(Some(&mut value)) == 0 {
            return Ok(());
        }
        let typed: Vec<LeftOut> = serde_json::from_value(value)?;
        conn.execute(
            "UPDATE main.delivery SET left_out_json = ?2 WHERE id = ?1",
            params![id, serde_json::to_string(&typed)?],
        )?;
        rewritten += 1;
        Ok(())
    })?;
    Ok((visited, rewritten))
}

const SLOTS: &str = "AND kind = 'slot'";

fn slots_earlier(conn: &Connection) -> Result<u64, Error> {
    let mut n = 0;
    each_row(conn, "annotation", "id", &["value"], SLOTS, |_, row| {
        if let Some(value) = text(&row[0]) {
            n += u64::from(slot(Some(&mut serde_json::from_str(value)?)) > 0);
        }
        Ok(())
    })?;
    Ok(n)
}

fn slots(conn: &Connection) -> Result<(u64, u64), Error> {
    let mut rewritten = 0;
    let visited = each_row(
        conn,
        "annotation",
        "id",
        &["key", "value"],
        SLOTS,
        |id, row| {
            let Some(stored) = text(&row[1]) else {
                return Ok(());
            };
            let mut value: Value = serde_json::from_str(stored)?;
            if slot(Some(&mut value)) == 0 {
                return Ok(());
            }
            polylogue_model::annotation::values::decode(
                polylogue_model::annotation::AnnotationKind::Slot,
                text(&row[0]),
                Some(&value),
            )?;
            conn.execute(
                "UPDATE main.annotation SET value = ?2 WHERE id = ?1",
                params![id, serde_json::to_string(&value)?],
            )?;
            rewritten += 1;
            Ok(())
        },
    )?;
    Ok((visited, rewritten))
}

/// Signal events have a key of two columns; they are read a delivery at a
/// time, in delivery order, each delivery's events whole.
fn each_signal_event(
    conn: &Connection,
    mut visit: impl FnMut(&str, i64, &[u8]) -> Result<(), Error>,
) -> Result<u64, Error> {
    let deliveries: Vec<String> = conn
        .prepare("SELECT DISTINCT delivery FROM main.signal_event ORDER BY delivery")?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut visited = 0;
    for delivery in deliveries {
        let events: Vec<(i64, Vec<u8>)> = conn
            .prepare_cached(
                "SELECT position, body FROM main.signal_event WHERE delivery = ?1 ORDER BY position",
            )?
            .query_map([&delivery], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (position, body) in events {
            visited += 1;
            visit(&delivery, position, &body)?;
        }
    }
    Ok(visited)
}

fn signal_events_earlier(conn: &Connection) -> Result<u64, Error> {
    let mut n = 0;
    each_signal_event(conn, |_, _, body| {
        n += u64::from(rename(&mut unpack(body)?, "ordinal", "request_position"));
        Ok(())
    })?;
    Ok(n)
}

fn signal_events(conn: &Connection) -> Result<(u64, u64), Error> {
    let mut rewritten = 0;
    let visited = each_signal_event(conn, |delivery, position, body| {
        let mut value = unpack(body)?;
        if rename(&mut value, "ordinal", "request_position") == 0 {
            return Ok(());
        }
        let typed: EventRecord = serde_json::from_value(value)?;
        conn.execute(
            "UPDATE main.signal_event SET body = ?3 WHERE delivery = ?1 AND position = ?2",
            params![delivery, position, pack(&typed)?],
        )?;
        rewritten += 1;
        Ok(())
    })?;
    Ok((visited, rewritten))
}

#[cfg(test)]
mod tests;
