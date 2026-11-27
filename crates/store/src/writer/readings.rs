//! A chain's reading in the reading tables (schema/index.sql, readings):
//! writing the rows a parse step states, and deleting a reading. Every row
//! carries its input and chain and holds what the chain's records say about
//! one thing, so writing or deleting a reading touches no other reading.
//! What a step changed is collected in [`Touched`]; the shown rows of those
//! things are computed afterwards from the readings present
//! ([`super::shown`]).
use super::*;
use polylogue_model::{Completion, IdKind, Route};
use std::collections::{BTreeMap, BTreeSet};

/// The things whose readings a transaction changed, by the names their
/// readings give them.
#[derive(Debug, Default)]
pub(crate) struct Touched {
    /// Sessions whose readings changed, as stated.
    pub sessions: BTreeSet<String>,
    /// Sessions to compose whole: a reading of theirs was replaced, or which
    /// session a copy's name names may have changed.
    pub whole: BTreeSet<String>,
    /// Messages, as (stated session, stated message ID).
    pub messages: BTreeSet<(String, String)>,
    /// Events, as (stated session, stated event ID).
    pub events: BTreeSet<(Option<String>, String)>,
    /// Link statements, as stated.
    pub links: BTreeSet<Vec<u8>>,
    pub documents: BTreeSet<String>,
    /// Calls whose background outcomes changed: (stated session, call ID).
    pub calls: BTreeSet<(String, String)>,
    /// Text rows a deleted reading row held; those no reading holds any
    /// more leave before the commit.
    pub released_texts: BTreeSet<i64>,
    /// Content groups a deleted or replaced reading row named; those nothing
    /// names any more leave before the commit, once the shown rows name the
    /// readings present.
    pub released_groups: BTreeSet<i64>,
}

/// The chain a step's rows belong to and what they are written with.
pub(crate) struct Chain<'a> {
    pub input_pk: i64,
    pub input: &'a str,
    pub route: Route,
    pub chain: i64,
    /// The record commit the rows were read at, kept with each row.
    pub committed: Option<i64>,
    /// The transaction's change summary: relations named by copied-prefix
    /// facts are recorded before they change.
    pub changes: &'a crate::change_summary::ChangeSummary,
    /// The chain gains or loses its parse with this step, which decides
    /// which content-named messages of the input's older chains are held.
    pub holds_changed: bool,
}

/// The coordinates of a link as its chain states them: its natural key.
pub(crate) fn link_statement(link: &Link) -> Result<Vec<u8>> {
    link.validate().map_err(|e| StoreError::Invalid(e.into()))?;
    Ok(serde_json::to_vec(&(
        link.parent.to_string(),
        link.child.to_string(),
        link.kind.as_str(),
        link.reason.map(|reason| reason.as_str()),
        link.branch_point.as_ref().map(ToString::to_string),
        link.call_id.as_deref(),
    ))?)
}

/// An event's ID: its content ID when it names a session; otherwise an ID
/// of its input record, since an event about its input is that record's.
pub(crate) fn event_id(
    input: &str,
    chain: i64,
    event: &Event,
    content_id: &str,
    occurrence: u64,
) -> Result<String> {
    if event.session.is_some() {
        return Ok(content_id.to_owned());
    }
    let (start, end) = event
        .range
        .filter(|(start, end)| start < end)
        .ok_or_else(|| {
            StoreError::Invalid("an event about its input has no record range".into())
        })?;
    if input.is_empty() || event.after.is_some() {
        return Err(StoreError::Invalid(
            "an input event needs its input identity and cannot follow a message".into(),
        ));
    }
    let mut hash = blake3::Hasher::new_derive_key("polylogue input event v1");
    hash.update(&(input.len() as u64).to_le_bytes());
    hash.update(input.as_bytes());
    hash.update(&chain.to_le_bytes());
    hash.update(&start.to_le_bytes());
    hash.update(&end.to_le_bytes());
    hash.update(&occurrence.to_le_bytes());
    hash.update(content_id.as_bytes());
    Ok(hash.finalize().to_hex().to_string())
}

fn position(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| StoreError::Invalid("a source position exceeds SQLite's integer range".into()))
}

/// The reading tables, and the per-chain tables that are part of a reading.
const READING_TABLES: &[&str] = &[
    "session_parse",
    "session_alias",
    "message_parse",
    "parse_text",
    "message_alias",
    "event_parse",
    "link_parse",
    "document_parse",
    "call_outcome",
    "message_injection_field",
    "shared_prefix_message",
    "shared_prefix",
    "unused",
    "export_parse",
    "repo_lookup",
];

/// Deletes a chain's reading: its rows in every reading table. What they
/// stated is recorded in `touched` so the shown rows are computed again, and
/// the texts they held are released.
pub(crate) fn delete(
    tx: &Connection,
    changes: &crate::change_summary::ChangeSummary,
    input_pk: i64,
    chain: i64,
    touched: &mut Touched,
) -> Result<()> {
    let key = params![input_pk, chain];
    let strings = |sql: &str| -> Result<Vec<String>> {
        Ok(tx
            .prepare_cached(sql)?
            .query_map(key, |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    };
    for session in strings(
        "SELECT session_id FROM session_parse WHERE input_pk=?1 AND chain=?2
         UNION SELECT session_id FROM message_parse WHERE input_pk=?1 AND chain=?2
         UNION SELECT session_id FROM event_parse WHERE input_pk=?1 AND chain=?2 AND session_id IS NOT NULL",
    )? {
        touched.whole.insert(session);
    }
    for alias in strings("SELECT alias FROM session_alias WHERE input_pk=?1 AND chain=?2")? {
        touch_copy(tx, &alias, touched)?;
    }
    for row in tx
        .prepare_cached(
            "SELECT session_id,message_id FROM message_parse WHERE input_pk=?1 AND chain=?2",
        )?
        .query_map(key, |row| <(String, String)>::try_from(row))?
    {
        touched.messages.insert(row?);
    }
    for row in tx
        .prepare_cached(
            "SELECT session_id,event_id FROM event_parse WHERE input_pk=?1 AND chain=?2",
        )?
        .query_map(key, |row| <(Option<String>, String)>::try_from(row))?
    {
        touched.events.insert(row?);
    }
    for row in tx
        .prepare_cached("SELECT statement FROM link_parse WHERE input_pk=?1 AND chain=?2")?
        .query_map(key, |row| row.get::<_, Vec<u8>>(0))?
    {
        touched.links.insert(row?);
    }
    touched.documents.extend(strings(
        "SELECT document_id FROM document_parse WHERE input_pk=?1 AND chain=?2",
    )?);
    for row in tx
        .prepare_cached(
            "SELECT session_id,call_id FROM call_outcome WHERE input_pk=?1 AND chain=?2",
        )?
        .query_map(key, |row| <(String, String)>::try_from(row))?
    {
        touched.calls.insert(row?);
    }
    for id in tx
        .prepare_cached("SELECT DISTINCT text_id FROM parse_text WHERE input_pk=?1 AND chain=?2")?
        .query_map(key, |row| row.get::<_, i64>(0))?
    {
        touched.released_texts.insert(id?);
    }
    for group in tx
        .prepare_cached(
            "SELECT content_group FROM message_parse WHERE input_pk=?1 AND chain=?2
             UNION SELECT content_group FROM event_parse WHERE input_pk=?1 AND chain=?2",
        )?
        .query_map(key, |row| row.get::<_, i64>(0))?
    {
        touched.released_groups.insert(group?);
    }
    // Copied-prefix facts name both sessions; either may read differently.
    for session in strings(
        "SELECT parent_id FROM shared_prefix WHERE input_pk=?1 AND chain=?2
         UNION SELECT child_id FROM shared_prefix WHERE input_pk=?1 AND chain=?2",
    )? {
        touched.sessions.insert(session);
    }
    for child in strings("SELECT child_id FROM shared_prefix WHERE input_pk=?1 AND chain=?2")? {
        changes.link_child(tx, &child)?;
    }
    crate::fragment::retract(tx, input_pk, chain)?;
    for table in READING_TABLES {
        tx.prepare_cached(&format!(
            "DELETE FROM {table} WHERE input_pk=?1 AND chain=?2"
        ))?
        .execute(key)?;
    }
    Ok(())
}

/// A copy's name gained or lost a recording session: the copy and every
/// session recording the name may now read differently.
fn touch_copy(tx: &Connection, alias: &str, touched: &mut Touched) -> Result<()> {
    touched.whole.insert(alias.to_owned());
    for session in tx
        .prepare_cached("SELECT DISTINCT session_id FROM session_alias WHERE alias=?1")?
        .query_map([alias], |row| row.get::<_, String>(0))?
    {
        touched.whole.insert(session?);
    }
    Ok(())
}

/// One stored reading of a message, as far as a step continuing the chain
/// needs it.
struct Stored {
    start: i64,
    hash: Vec<u8>,
    layout: Vec<u8>,
    group: i64,
}

/// Writes the rows one parse step states into its chain's reading.
pub(crate) fn write(
    tx: &Transaction<'_>,
    chain: &Chain<'_>,
    batch: &Batch,
    step: &ParsePart,
    touched: &mut Touched,
) -> Result<()> {
    let key = params![chain.input_pk, chain.chain];
    let mut groups = crate::content::Stored::new(&step.prepared.content);
    write_sessions(tx, chain, step, touched)?;
    write_messages(tx, chain, step, &mut groups, touched)?;
    write_documents(tx, chain, step, touched)?;
    write_events(tx, chain, step, &mut groups, touched)?;
    for alias in &step.output.aliases {
        tx.prepare_cached(
            "INSERT INTO message_alias(input_pk,chain,name,target,source_start,source_length) VALUES (?1,?2,?3,?4,?5,?6-?5)
             ON CONFLICT(input_pk,chain,name) DO UPDATE SET target=excluded.target,source_start=excluded.source_start,source_length=excluded.source_length",
        )?
        .execute(params![
            chain.input_pk,
            chain.chain,
            alias.name.to_string(),
            alias.target.as_ref().map(ToString::to_string),
            position(alias.offset)?,
            position(alias.end)?
        ])?;
    }
    write_links(tx, chain, step, touched)?;
    for prefix in &step.output.prefix_copies {
        write_prefix_copy(
            tx,
            chain,
            &prefix.parent.to_string(),
            &prefix.child.to_string(),
            prefix.explicit_members,
            touched,
        )?;
    }
    for member in &step.output.prefix_members {
        let (parent, child) = (member.parent.to_string(), member.child.to_string());
        let source = member
            .source
            .clone()
            .unwrap_or_else(|| polylogue_model::PrefixSource {
                input: chain.input.to_owned(),
                chain: chain.chain,
                parse_version: step.parse_version.clone(),
            });
        tx.prepare_cached("INSERT INTO shared_prefix_message(input_pk,chain,parent_id,child_id,message_id,source_input,source_chain,source_version,position,record_position,committed) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11) ON CONFLICT DO UPDATE SET committed=coalesce(max(shared_prefix_message.committed,excluded.committed),shared_prefix_message.committed,excluded.committed)")?
            .execute(params![chain.input_pk,chain.chain,parent,child,member.message.to_string(),source.input,source.chain,source.parse_version,member.position[0].to_string(),member.position[1].to_string(),chain.committed])?;
        write_prefix_copy(tx, chain, &parent, &child, true, touched)?;
    }
    if batch
        .input
        .route
        .lists_every_conversation(&batch.input.format)
    {
        let mut latest = HashMap::<&str, i64>::new();
        for row in &step.output.messages {
            if let Some(at) = row.message.at.map(Timestamp::unix_ms) {
                let origin = latest
                    .entry(row.message.session.origin().as_str())
                    .or_insert(at);
                *origin = (*origin).max(at);
            }
        }
        for (origin, at) in latest {
            tx.prepare_cached(
                "INSERT INTO export_parse(input_pk,chain,origin,latest_at) VALUES (?1,?2,?3,?4)
                 ON CONFLICT(input_pk,chain,origin) DO UPDATE SET latest_at=max(latest_at,excluded.latest_at)",
            )?
            .execute(params![chain.input_pk, chain.chain, origin, at])?;
        }
    }
    for lookup in &step.output.repo_lookups {
        crate::repo_lookup::upsert_lookup(tx, lookup, chain.input_pk, chain.chain)?;
    }
    for (n, update) in step.output.background.iter().enumerate() {
        let Some(call_id) = update.call_id.as_deref() else {
            continue;
        };
        let session = update.session.to_string();
        let next: i64 = tx.scalar(
            "SELECT coalesce(max(position)+1,0) FROM call_outcome WHERE input_pk=?1 AND chain=?2",
            key,
        )?;
        tx.prepare_cached(
            "INSERT INTO call_outcome(input_pk,chain,session_id,call_id,position,body) VALUES (?1,?2,?3,?4,?5,?6)
             ON CONFLICT(input_pk,chain,session_id,call_id) DO UPDATE SET position=excluded.position,body=excluded.body",
        )?
        .execute(params![
            chain.input_pk,
            chain.chain,
            session,
            call_id,
            next + n as i64,
            serde_json::to_string(update)?
        ])?;
        touched.calls.insert((session, call_id.to_owned()));
    }
    step.prepared
        .fragments
        .apply(tx, chain.input_pk, chain.chain, step.parsed_upto, |_| {
            Ok(chain.committed)
        })?;
    if chain.holds_changed {
        // Once a newer chain is parsed, an older chain's content-named
        // messages that it no longer holds are gone (`held_message_parse`).
        for row in tx
            .prepare_cached(
                "SELECT session_id,message_id FROM message_parse
                 WHERE input_pk=?1 AND chain<?2 AND id_kind='content'",
            )?
            .query_map(key, |row| <(String, String)>::try_from(row))?
        {
            touched.messages.insert(row?);
        }
    }
    Ok(())
}

fn write_prefix_copy(
    tx: &Connection,
    chain: &Chain<'_>,
    parent: &str,
    child: &str,
    explicit_members: bool,
    touched: &mut Touched,
) -> Result<()> {
    chain.changes.link_child(tx, child)?;
    tx.prepare_cached(
        "INSERT INTO shared_prefix(input_pk,chain,parent_id,child_id,committed,explicit_members) VALUES (?1,?2,?3,?4,?5,?6)
         ON CONFLICT DO UPDATE SET explicit_members=max(explicit_members,excluded.explicit_members),
         committed=coalesce(max(shared_prefix.committed,excluded.committed),shared_prefix.committed,excluded.committed)",
    )?
    .execute(params![chain.input_pk, chain.chain, parent, child, chain.committed, explicit_members])?;
    touched.sessions.insert(parent.to_owned());
    touched.sessions.insert(child.to_owned());
    Ok(())
}

fn write_sessions(
    tx: &Connection,
    chain: &Chain<'_>,
    step: &ParsePart,
    touched: &mut Touched,
) -> Result<()> {
    for session in &step.output.sessions {
        let id = session.id.to_string();
        let stored: Option<String> = tx.scalar_optional(
            "SELECT facts FROM session_parse WHERE input_pk=?1 AND chain=?2 AND session_id=?3",
            params![chain.input_pk, chain.chain, id],
        )?;
        let stored: Option<Session> = stored
            .map(|facts| serde_json::from_str(&facts))
            .transpose()?;
        let mut merged = stored.clone().unwrap_or_else(|| session.clone());
        merged.merge(session);
        tx.prepare_cached(
            "INSERT INTO session_parse(input_pk,chain,session_id,facts,committed) VALUES (?1,?2,?3,?4,?5)
             ON CONFLICT(input_pk,chain,session_id) DO UPDATE SET facts=excluded.facts,
             committed=coalesce(max(session_parse.committed,excluded.committed),session_parse.committed,excluded.committed)",
        )?
        .execute(params![chain.input_pk, chain.chain, id, serde_json::to_string(&merged)?, chain.committed])?;
        let before: BTreeSet<String> = stored
            .iter()
            .flat_map(|s| s.aliases.iter().map(ToString::to_string))
            .collect();
        let after: BTreeSet<String> = merged.aliases.iter().map(ToString::to_string).collect();
        if before != after {
            tx.prepare_cached(
                "DELETE FROM session_alias WHERE input_pk=?1 AND chain=?2 AND session_id=?3",
            )?
            .execute(params![chain.input_pk, chain.chain, id])?;
            for alias in &after {
                tx.prepare_cached(
                    "INSERT INTO session_alias(input_pk,chain,session_id,alias) VALUES (?1,?2,?3,?4)",
                )?
                .execute(params![chain.input_pk, chain.chain, id, alias])?;
            }
            for alias in before.symmetric_difference(&after) {
                touch_copy(tx, alias, touched)?;
            }
        }
        // The session's kind decides whether its messages' readings come from
        // a main session and whether it was explicitly recorded.
        if stored
            .as_ref()
            .is_some_and(|stored| stored.kind != merged.kind)
        {
            for row in tx
                .prepare_cached(
                    "SELECT message_id FROM message_parse WHERE input_pk=?1 AND chain=?2 AND session_id=?3",
                )?
                .query_map(params![chain.input_pk, chain.chain, id], |row| row.get::<_, String>(0))?
            {
                touched.messages.insert((id.clone(), row?));
            }
        }
        touched.sessions.insert(id);
    }
    Ok(())
}

/// Resolves the texts a body holds and writes them as the item's reading
/// texts, releasing the ones it no longer holds.
fn write_texts(
    tx: &Connection,
    chain: &Chain<'_>,
    document: bool,
    item: &str,
    texts: &[Option<[u8; 32]>],
    packed: &crate::text_rows::PreparedTexts,
    touched: &mut Touched,
) -> Result<()> {
    let key = params![chain.input_pk, chain.chain, document, item];
    for id in tx
        .prepare_cached(
            "SELECT text_id FROM parse_text WHERE input_pk=?1 AND chain=?2 AND document=?3 AND item=?4",
        )?
        .query_map(key, |row| row.get::<_, i64>(0))?
    {
        touched.released_texts.insert(id?);
    }
    tx.prepare_cached(
        "DELETE FROM parse_text WHERE input_pk=?1 AND chain=?2 AND document=?3 AND item=?4",
    )?
    .execute(key)?;
    for (idx, hash) in texts.iter().enumerate() {
        let Some(hash) = hash else { continue };
        let id = crate::text_rows::resolve(tx, hash, packed.get(hash))?;
        tx.prepare_cached(
            "INSERT INTO parse_text(input_pk,chain,document,item,position,text_id) VALUES (?1,?2,?3,?4,?5,?6)",
        )?
        .execute(params![chain.input_pk, chain.chain, document, item, idx as i64, id])?;
    }
    Ok(())
}

fn write_messages(
    tx: &Connection,
    chain: &Chain<'_>,
    step: &ParsePart,
    groups: &mut crate::content::Stored<'_>,
    touched: &mut Touched,
) -> Result<()> {
    // Messages one record holds share its offset; their order in the record
    // breaks the tie.
    let mut in_record = BTreeMap::<u64, i64>::new();
    // The step's kept readings by session, with their places.
    let mut kept = BTreeMap::<String, Vec<(i64, i64)>>::new();
    for (i, (row, prepared)) in step
        .output
        .messages
        .iter()
        .zip(&step.prepared.messages)
        .enumerate()
    {
        let id = row.message.id.to_string();
        let session = row.message.session.to_string();
        let place = match step.message_position(i)? {
            Some(order) => (order, 0),
            None => {
                let next = in_record.entry(row.offset).or_default();
                *next += 1;
                (position(row.offset)?, *next - 1)
            }
        };
        let stored: Option<Stored> = tx
            .prepare_cached(
                "SELECT source_start,hash,layout,content_group FROM message_parse WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
            )?
            .query_row(params![chain.input_pk, chain.chain, id], |r| {
                Ok(Stored {
                    start: r.get(0)?,
                    hash: r.get(1)?,
                    layout: r.get(2)?,
                    group: r.get(3)?,
                })
            })
            .optional()?;
        let offset = position(row.offset)?;
        // A chain states a message where it first states it; one that
        // restates whole conversations, where it last does.
        let keep = !stored.as_ref().is_some_and(|stored| {
            if step.repeats_whole_conversations {
                offset < stored.start
            } else {
                offset > stored.start
            }
        });
        // Statements of one version that each select some of its blocks
        // are one reading holding every block any of them selected.
        let merged = match &stored {
            Some(stored)
                if stored.hash == prepared.hash
                    && (prepared.block_selection.block_count != prepared.texts.len() as u64
                        || stored.layout
                            != crate::encoding::message_layout(&prepared.block_selection)?) =>
            {
                let previous = read_message(tx, chain.input_pk, chain.chain, &id)?
                    .ok_or_else(|| StoreError::Invalid("a stored reading has no body".into()))?;
                let mut incoming = row.message.clone();
                let stored_value = step.prepared.message_value(i)?;
                crate::writer::attachments::restore_values(
                    &mut incoming.blocks,
                    &stored_value.blocks,
                )?;
                let (message, selection) = previous
                    .1
                    .merge(&previous.0, &prepared.block_selection, &incoming)?
                    .ok_or(polylogue_model::bundle::BundleError::SourceVersionMismatch)?;
                Some((message, selection))
            }
            _ => None,
        };
        if !keep && merged.is_none() {
            continue;
        }
        // A merge is a new value: its content and texts are prepared here.
        let with_merged;
        let (prepared, packed, content) = match merged {
            Some((mut message, selection)) => {
                let mut packer = crate::content::Packer::default();
                let stored = step
                    .output
                    .stored_fields
                    .get(&i)
                    .map_or(&[][..], Vec::as_slice);
                let prepared =
                    PreparedMessage::projected(&mut message, &selection, stored, &mut packer)?;
                let groups = packer.finish()?;
                let content = crate::content::Stored::new(&groups).place(tx, prepared.content)?;
                let mut packed = step.prepared.texts.clone();
                packed.add(
                    message
                        .blocks
                        .iter()
                        .filter_map(crate::encoding::block_text),
                )?;
                with_merged = (prepared, packed);
                (&with_merged.0, &with_merged.1, content)
            }
            None => (
                prepared,
                &step.prepared.texts,
                groups.place(tx, prepared.content)?,
            ),
        };
        write_texts(tx, chain, false, &id, &prepared.texts, packed, touched)?;
        let layout = crate::encoding::message_layout(&prepared.block_selection)?;
        if let Some(stored) = &stored
            && stored.group != content.group
        {
            touched.released_groups.insert(stored.group);
        }
        if keep {
            let route = row.route(chain.route);
            let message = &row.message;
            tx.prepare_cached(
                "INSERT INTO message_parse(input_pk,chain,message_id,session_id,source_start,source_length,position,record_position,
                     at,route,state,id_kind,hash,layout,response_key,content_name,
                     content_group,content_start,content_length)
                 VALUES (?1,?2,?3,?4,?5,?6-?5,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)
                 ON CONFLICT(input_pk,chain,message_id) DO UPDATE SET session_id=excluded.session_id,
                     source_start=excluded.source_start,source_length=excluded.source_length,position=excluded.position,
                     record_position=excluded.record_position,at=excluded.at,route=excluded.route,
                     state=excluded.state,id_kind=excluded.id_kind,hash=excluded.hash,
                     layout=excluded.layout,response_key=excluded.response_key,content_name=excluded.content_name,
                     content_group=excluded.content_group,
                     content_start=excluded.content_start,content_length=excluded.content_length",
            )?
            .execute(params![
                chain.input_pk,
                chain.chain,
                id,
                session,
                offset,
                position(row.end.max(row.offset))?,
                place.0,
                place.1,
                message.at.map(Timestamp::unix_ms),
                route.as_str(),
                Completion::of(message, route).as_str(),
                if row.id_kind == IdKind::Content { "content" } else { "native" },
                prepared.hash.as_slice(),
                layout,
                row.response,
                crate::chain_names::content_name(row),
                content.group,
                content.start,
                content.length,
            ])?;
            kept.entry(session.clone()).or_default().push(place);
        } else {
            tx.prepare_cached(
                "UPDATE message_parse SET layout=?4,content_group=?5,content_start=?6,content_length=?7
                 WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
            )?
            .execute(params![chain.input_pk, chain.chain, id, layout, content.group, content.start, content.length])?;
        }
        write_injection_fields(tx, chain, step, i, &id)?;
        touched.messages.insert((session, id));
    }
    for (session, mut places) in kept {
        places.sort();
        order_times(
            tx,
            chain,
            &session,
            places[0],
            step.repeats_whole_conversations,
            touched,
        )?;
    }
    Ok(())
}

/// Gives the chain's readings in `session` from `first` on their order
/// times: the latest source time of each and the readings before it in the
/// chain and session. A step that restates whole conversations starts
/// afresh; any other continues from the reading before its first, so an
/// append reads only its own rows, and a step that reads an earlier message
/// again with another time carries the change through the chain's later
/// readings, as reading the chain in one step would.
fn order_times(
    tx: &Connection,
    chain: &Chain<'_>,
    session: &str,
    first: (i64, i64),
    restates: bool,
    touched: &mut Touched,
) -> Result<()> {
    let mut running: Option<i64> = if restates {
        None
    } else {
        tx.prepare_cached(
            "SELECT order_time FROM message_parse WHERE input_pk=?1 AND chain=?2 AND session_id=?3
             AND (position,record_position)<(?4,?5) ORDER BY position DESC,record_position DESC LIMIT 1",
        )?
        .query_row(params![chain.input_pk, chain.chain, session, first.0, first.1], |r| r.get(0))
        .optional()?
        .flatten()
    };
    let later = tx
        .prepare_cached(
            "SELECT message_id,at,order_time FROM message_parse WHERE input_pk=?1 AND chain=?2 AND session_id=?3
             AND (position,record_position)>=(?4,?5) ORDER BY position,record_position",
        )?
        .query_map(params![chain.input_pk, chain.chain, session, first.0, first.1], |r| {
            <(String, Option<i64>, Option<i64>)>::try_from(r)
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (id, at, stored) in later {
        running = running.max(at);
        if running != stored {
            tx.prepare_cached(
                "UPDATE message_parse SET order_time=?4 WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
            )?
            .execute(params![chain.input_pk, chain.chain, id, running])?;
            touched.messages.insert((session.to_owned(), id));
        }
    }
    Ok(())
}

fn write_injection_fields(
    tx: &Connection,
    chain: &Chain<'_>,
    step: &ParsePart,
    index: usize,
    message: &str,
) -> Result<()> {
    use polylogue_model::injection::StatedHookField;
    tx.prepare_cached(
        "DELETE FROM message_injection_field WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
    )?
    .execute(params![chain.input_pk, chain.chain, message])?;
    let insert = |block: u64, field: StatedHookField| -> Result<()> {
        field.validate()?;
        tx.prepare_cached(
            "INSERT OR IGNORE INTO message_injection_field(input_pk,chain,message_id,source_block_position,field) VALUES (?1,?2,?3,?4,?5)",
        )?
        .execute(params![chain.input_pk, chain.chain, message, block.to_string(), serde_json::to_string(&field)?])?;
        Ok(())
    };
    if let Some(stated) = step.output.hook_fields.get(&index) {
        for stated_field in stated {
            insert(
                stated_field.block,
                StatedHookField {
                    input: chain.input.into(),
                    chain: chain.chain,
                    start: stated_field.start,
                    end: stated_field.end,
                    pointer: stated_field.pointer.clone(),
                },
            )?;
        }
    }
    if let Some(injections) = step.output.declared_injections.get(&index) {
        for injection in injections {
            // A forged observed member stays a declaration from another source
            // even if an importer skipped its into_declared conversion.
            let declared = injection.clone().into_declared();
            for field in declared.declared {
                insert(declared.block, field)?;
            }
        }
    }
    Ok(())
}

fn write_documents(
    tx: &Connection,
    chain: &Chain<'_>,
    step: &ParsePart,
    touched: &mut Touched,
) -> Result<()> {
    for (row, prepared) in step.output.documents.iter().zip(&step.prepared.documents) {
        let id = row.document.id.to_string();
        let time = row.document.source_time;
        let stored: Option<(Option<i64>, String, Vec<u8>)> = tx
            .prepare_cached(
                "SELECT at,route,hash FROM document_parse WHERE input_pk=?1 AND chain=?2 AND document_id=?3",
            )?
            .query_row(params![chain.input_pk, chain.chain, id], |r| <(_, _, _)>::try_from(r))
            .optional()?;
        // A chain that holds the document twice keeps the reading the
        // document choice prefers between them.
        if let Some((stored_time, stored_route, stored_hash)) = &stored {
            let stored_route = Route::from_name(stored_route)
                .ok_or_else(|| StoreError::stored(0, "unknown stored route"))?;
            let stored_time = stored_time.and_then(Timestamp::from_unix_ms);
            let readings = [
                compose::DocumentReading {
                    input: chain.input,
                    time: stored_time,
                    route: stored_route,
                    hash: stored_hash,
                },
                compose::DocumentReading {
                    input: chain.input,
                    time,
                    route: chain.route,
                    hash: &prepared.hash,
                },
            ];
            if compose::document(&readings) == Some(0) && stored_hash.as_slice() != prepared.hash {
                continue;
            }
        }
        write_texts(
            tx,
            chain,
            true,
            &id,
            &prepared.texts,
            &step.prepared.texts,
            touched,
        )?;
        tx.prepare_cached(
            "INSERT INTO document_parse(input_pk,chain,document_id,route,source_start,source_length,at,at_details,hash,body)
             VALUES (?1,?2,?3,?4,?5,?6-?5,?7,?8,?9,?10)
             ON CONFLICT(input_pk,chain,document_id) DO UPDATE SET route=excluded.route,source_start=excluded.source_start,
                 source_length=excluded.source_length,at=excluded.at,at_details=excluded.at_details,hash=excluded.hash,
                 body=excluded.body",
        )?
        .execute(params![
            chain.input_pk,
            chain.chain,
            id,
            chain.route.as_str(),
            position(row.offset)?,
            position(row.end)?,
            time.map(Timestamp::unix_ms),
            prepared.time,
            prepared.hash.as_slice(),
            prepared.body,
        ])?;
        touched.documents.insert(id);
    }
    Ok(())
}

fn write_events(
    tx: &Connection,
    chain: &Chain<'_>,
    step: &ParsePart,
    groups: &mut crate::content::Stored<'_>,
    touched: &mut Touched,
) -> Result<()> {
    if step.output.events.is_empty() {
        return Ok(());
    }
    let ids = step.event_identities(chain.input, chain.chain)?;
    let mut reader = crate::content::Reader::default();
    for (i, (event, prepared)) in step
        .output
        .events
        .iter()
        .zip(&step.prepared.events)
        .enumerate()
    {
        let id = &ids[i];
        let imported = step.output.imported_events.get(&i);
        let seq = step
            .order_from
            .map(|n| n as i64)
            .or(imported
                .is_none()
                .then_some(event.range)
                .flatten()
                .map(|r| r.0 as i64))
            .or(step
                .repeats_whole_conversations
                .then_some(step.start as i64));
        let content = groups.place(tx, prepared.content)?;
        let fields = crate::content::Stretch {
            length: prepared.fields_length as i64,
            ..content
        };
        let mut original = imported
            .map(|identity| identity.sources.clone())
            .unwrap_or_default();
        // A chain stating the same event again with the same value keeps
        // the sources both statements imported.
        let stored: Option<(Vec<u8>, crate::content::Stretch, String)> = tx
            .prepare_cached(
                "SELECT hash,content_group,content_start,content_length,imported_sources FROM event_parse
                 WHERE input_pk=?1 AND chain=?2 AND event_id=?3",
            )?
            .query_row(params![chain.input_pk, chain.chain, id], |r| {
                Ok((
                    r.get(0)?,
                    crate::content::Stretch {
                        group: r.get(1)?,
                        start: r.get(2)?,
                        length: r.get(3)?,
                    },
                    r.get(4)?,
                ))
            })
            .optional()?;
        if let Some((stored_hash, stored_fields, stored_imported)) = stored {
            if stored_fields.group != content.group {
                touched.released_groups.insert(stored_fields.group);
            }
            if stored_hash == prepared.hash
                && reader.bytes(tx, stored_fields)?.to_vec() == reader.bytes(tx, fields)?
            {
                original.extend(serde_json::from_str::<
                    Vec<polylogue_model::bundle::StatedSource>,
                >(&stored_imported)?);
            }
        }
        original.sort();
        original.dedup();
        let range = if imported.is_some() {
            Some((step.start, step.parsed_upto))
        } else {
            event.range.or(Some((step.start, step.parsed_upto)))
        };
        let session = event.session.as_ref().map(ToString::to_string);
        tx.prepare_cached(
            "INSERT INTO event_parse(input_pk,chain,event_id,session_id,route,position,source_start,source_length,
                 after_message_id,kind,at,hash,imported_sources,
                 content_group,content_start,content_length,data_length)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8-?7,?9,?10,?11,?12,?13,?14,?15,?16,?17)
             ON CONFLICT(input_pk,chain,event_id) DO UPDATE SET session_id=excluded.session_id,route=excluded.route,
                 position=excluded.position,source_start=excluded.source_start,source_length=excluded.source_length,
                 after_message_id=excluded.after_message_id,kind=excluded.kind,at=excluded.at,hash=excluded.hash,
                 imported_sources=excluded.imported_sources,
                 content_group=excluded.content_group,content_start=excluded.content_start,
                 content_length=excluded.content_length,data_length=excluded.data_length",
        )?
        .execute(params![
            chain.input_pk,
            chain.chain,
            id,
            session,
            chain.route.as_str(),
            seq.unwrap_or(-1),
            range.map(|r| r.0 as i64),
            range.map(|r| r.1 as i64),
            event.after.as_ref().map(ToString::to_string),
            event.kind,
            event.at.map(Timestamp::unix_ms),
            prepared.hash.as_slice(),
            serde_json::to_string(&original)?,
            content.group,
            content.start,
            prepared.fields_length as i64,
            prepared.data_length() as i64,
        ])?;
        touched.events.insert((session, id.clone()));
    }
    Ok(())
}

fn write_links(
    tx: &Connection,
    chain: &Chain<'_>,
    step: &ParsePart,
    touched: &mut Touched,
) -> Result<()> {
    for link in &step.output.links {
        let statement = link_statement(link)?;
        let stored: Option<Vec<u8>> = tx.scalar_optional(
            "SELECT body FROM link_parse WHERE input_pk=?1 AND chain=?2 AND statement=?3",
            params![chain.input_pk, chain.chain, statement],
        )?;
        let mut merged = link.clone();
        if let Some(stored) = stored {
            let stored: Link = serde_json::from_slice(&stored)?;
            merged = merge_link(merged, stored);
        }
        tx.prepare_cached(
            "INSERT INTO link_parse(input_pk,chain,statement,parent_id,child_id,kind,reason,route,hash,body,committed)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
             ON CONFLICT(input_pk,chain,statement) DO UPDATE SET route=excluded.route,hash=excluded.hash,
                 body=excluded.body,committed=coalesce(max(link_parse.committed,excluded.committed),link_parse.committed,excluded.committed)",
        )?
        .execute(params![
            chain.input_pk,
            chain.chain,
            statement,
            link.parent.to_string(),
            link.child.to_string(),
            link.kind.as_str(),
            link.reason.map(|reason| reason.as_str()),
            chain.route.as_str(),
            crate::encoding::value_hash(&merged)?.as_slice(),
            serde_json::to_vec(&merged)?,
            chain.committed,
        ])?;
        touched.links.insert(statement);
    }
    Ok(())
}

/// Observations of the same coordinates combine; disagreements have other
/// statements and stay separate.
pub(crate) fn merge_link(mut link: Link, other: Link) -> Link {
    link.copies_prefix |= other.copies_prefix;
    link.evidence = link.evidence.min(other.evidence);
    link
}

/// A message reading as stored: its value with texts and structured values
/// restored, its block layout, and where each block's piece lies.
pub(crate) struct StoredReading {
    pub message: Message,
    pub selection: polylogue_model::bundle::ShownBlocks,
    pub pieces: Vec<crate::content::Piece>,
}

/// A message reading's value and block layout, with its texts restored.
pub(crate) fn read_message(
    conn: &Connection,
    input_pk: i64,
    chain: i64,
    message: &str,
) -> Result<Option<(Message, polylogue_model::bundle::ShownBlocks)>> {
    Ok(read_reading(
        conn,
        &mut crate::content::Reader::default(),
        input_pk,
        chain,
        message,
    )?
    .map(|stored| (stored.message, stored.selection)))
}

/// A message reading as stored ([`StoredReading`]).
pub(crate) fn read_reading(
    conn: &Connection,
    reader: &mut crate::content::Reader,
    input_pk: i64,
    chain: i64,
    message: &str,
) -> Result<Option<StoredReading>> {
    let stored: Option<(Vec<u8>, Vec<u8>, crate::content::Stretch)> = conn
        .prepare_cached(
            "SELECT layout,hash,content_group,content_start,content_length FROM message_parse
             WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
        )?
        .query_row(params![input_pk, chain, message], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                crate::content::Stretch {
                    group: r.get(2)?,
                    start: r.get(3)?,
                    length: r.get(4)?,
                },
            ))
        })
        .optional()?;
    let Some((layout, hash, content)) = stored else {
        return Ok(None);
    };
    let id = message;
    let (mut message, pieces) =
        crate::encoding::read_message_content(reader.bytes(conn, content)?, true)?;
    restore_texts(conn, input_pk, chain, false, id, &mut message.blocks)?;
    let selection =
        crate::encoding::message_block_selection(&hash, Some(&layout), message.blocks.len())?;
    selection.validate(message.blocks.len())?;
    Ok(Some(StoredReading {
        message,
        selection,
        pieces: placed(content, &pieces),
    }))
}

/// Where each block's piece lies in the content group of `content`.
pub(crate) fn placed(
    content: crate::content::Stretch,
    pieces: &[crate::encoding::BlockPiece],
) -> Vec<crate::content::Piece> {
    pieces
        .iter()
        .map(|piece| crate::content::Piece {
            group: content.group,
            start: content.start + piece.start as i64,
            length: piece.length as i64,
            data_length: piece.data_length.map(|n| n as i64),
        })
        .collect()
}

/// A document reading's body, with its texts restored.
pub(crate) fn read_document(
    conn: &Connection,
    input_pk: i64,
    chain: i64,
    document: &str,
) -> Result<Option<polylogue_model::Document>> {
    let body: Option<Vec<u8>> = conn.scalar_optional(
        "SELECT body FROM document_parse WHERE input_pk=?1 AND chain=?2 AND document_id=?3",
        params![input_pk, chain, document],
    )?;
    body.map(|body| {
        let mut read: polylogue_model::Document = crate::encoding::unpack_json(&body)?;
        restore_texts(conn, input_pk, chain, true, document, &mut read.blocks)?;
        Ok(read)
    })
    .transpose()
}

/// The text keys of a reading's message or document body, by block.
pub(crate) fn text_ids(
    conn: &Connection,
    input_pk: i64,
    chain: i64,
    document: bool,
    item: &str,
) -> Result<BTreeMap<usize, i64>> {
    Ok(conn
        .prepare_cached(
            "SELECT position,text_id FROM parse_text WHERE input_pk=?1 AND chain=?2 AND document=?3 AND item=?4",
        )?
        .query_map(params![input_pk, chain, document, item], |row| {
            Ok((row.get::<_, i64>(0)? as usize, row.get(1)?))
        })?
        .collect::<rusqlite::Result<_>>()?)
}

fn restore_texts(
    conn: &Connection,
    input_pk: i64,
    chain: i64,
    document: bool,
    item: &str,
    blocks: &mut [Block],
) -> Result<()> {
    let mut statement = conn.prepare_cached(
        "SELECT r.position,g.body,t.start,t.length FROM parse_text r JOIN text t ON t.id=r.text_id
         JOIN content_group g ON g.id=t.group_id
         WHERE r.input_pk=?1 AND r.chain=?2 AND r.document=?3 AND r.item=?4 ORDER BY r.position",
    )?;
    let mut rows = statement.query(params![input_pk, chain, document, item])?;
    while let Some(row) = rows.next()? {
        let idx: i64 = row.get(0)?;
        let text = crate::text_rows::joined(row, 1)?;
        let block = usize::try_from(idx)
            .ok()
            .and_then(|idx| blocks.get_mut(idx))
            .ok_or_else(|| StoreError::stored(0, "a stored text names a block the body lacks"))?;
        crate::encoding::restore_text(block, text);
    }
    Ok(())
}

/// Removes the released content groups nothing names any more.
pub(crate) fn remove_released_groups(tx: &Connection, released: &BTreeSet<i64>) -> Result<()> {
    for &group in released {
        crate::content::release(tx, group)?;
    }
    Ok(())
}

/// Removes the released texts no reading holds any more, with their search
/// entries.
pub(crate) fn remove_released_texts(tx: &Connection, released: &BTreeSet<i64>) -> Result<()> {
    for &id in released {
        let unheld: bool = tx
            .prepare_cached(
                "SELECT EXISTS(SELECT 1 FROM text t WHERE t.id=?1
                     AND NOT EXISTS(SELECT 1 FROM parse_text r WHERE r.text_id=t.id))",
            )?
            .query_row([id], |r| r.get(0))?;
        if unheld {
            crate::text_rows::remove(tx, id)?;
        }
    }
    Ok(())
}

/// The reading a shown message shows: its main reading's input, chain and
/// the message's ID there (a copy's message is named otherwise in its
/// session).
pub(crate) fn shown_reading(
    conn: &Connection,
    message_pk: i64,
) -> Result<Option<(i64, i64, String)>> {
    Ok(conn
        .prepare_cached(
            "SELECT r.input_pk,r.chain,r.message_id FROM message m JOIN message_parse r
             ON r.input_pk=m.input_pk AND r.chain=m.source_chain AND r.source_start=m.source_start
                AND r.hash=m.version_hash
             WHERE m.pk=?1 ORDER BY r.message_id=m.id DESC LIMIT 1",
        )?
        .query_row([message_pk], |row| <(i64, i64, String)>::try_from(row))
        .optional()?)
}

/// The value a shown message shows: its version as its readings hold it.
pub(crate) fn shown_message(
    conn: &Connection,
    message_pk: i64,
) -> Result<Option<(Message, polylogue_model::bundle::ShownBlocks)>> {
    let Some((input, chain, id)) = shown_reading(conn, message_pk)? else {
        return Ok(None);
    };
    let hash: Vec<u8> = conn
        .prepare_cached(
            "SELECT hash FROM message_parse WHERE input_pk=?1 AND chain=?2 AND message_id=?3",
        )?
        .query_row(params![input, chain, id], |row| row.get(0))?;
    let shown: String = conn
        .prepare_cached("SELECT id FROM message WHERE pk=?1")?
        .query_row([message_pk], |row| row.get(0))?;
    let mut ids = vec![id];
    if shown != ids[0] {
        ids.push(shown);
    }
    read_version(conn, &ids, &hash)
}

/// A message version as its readings hold it: readings of one version that
/// each hold some of its blocks (sparse exports) together hold every block
/// any of them holds. `ids` are the message's IDs as its readings name it.
pub(crate) fn read_version(
    conn: &Connection,
    ids: &[String],
    hash: &[u8],
) -> Result<Option<(Message, polylogue_model::bundle::ShownBlocks)>> {
    Ok(
        read_version_reading(conn, &mut crate::content::Reader::default(), ids, hash)?
            .map(|stored| (stored.message, stored.selection)),
    )
}

/// [`read_version`], with where each block's piece lies: each block is named
/// in the content of a reading that holds it.
pub(crate) fn read_version_reading(
    conn: &Connection,
    reader: &mut crate::content::Reader,
    ids: &[String],
    hash: &[u8],
) -> Result<Option<StoredReading>> {
    let mut found = Vec::new();
    for id in ids {
        let rows = conn
            .prepare_cached(
                "SELECT i.id,r.input_pk,r.chain FROM message_parse r JOIN input_ref i ON i.pk=r.input_pk
                 WHERE r.message_id=?1 AND r.hash=?2",
            )?
            .query_map(params![id, hash], |row| <(String, i64, i64)>::try_from(row))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        found.extend(
            rows.into_iter()
                .map(|(input, pk, chain)| (input, chain, pk, id.clone())),
        );
    }
    found.sort();
    let mut merged: Option<(Message, polylogue_model::bundle::ShownBlocks)> = None;
    let mut pieces = HashMap::<u64, crate::content::Piece>::new();
    for (_, chain, input_pk, id) in found {
        let Some(stored) = read_reading(conn, reader, input_pk, chain, &id)? else {
            continue;
        };
        for (ordinal, piece) in stored.selection.block_ordinals.iter().zip(&stored.pieces) {
            pieces.entry(*ordinal).or_insert(*piece);
        }
        let (message, selection) = (stored.message, stored.selection);
        let complete = selection.block_count == message.blocks.len() as u64;
        merged = Some(match merged {
            None => (message, selection),
            Some(_) if complete => (message, selection),
            Some((held, held_selection)) => {
                if held_selection.block_count == held.blocks.len() as u64 {
                    (held, held_selection)
                } else {
                    held_selection
                        .merge(&held, &selection, &message)?
                        .ok_or(polylogue_model::bundle::BundleError::SourceVersionMismatch)?
                }
            }
        });
        if let Some((held, held_selection)) = &merged
            && held_selection.block_count == held.blocks.len() as u64
        {
            break;
        }
    }
    merged
        .map(|(message, selection)| {
            let pieces = selection
                .block_ordinals
                .iter()
                .map(|ordinal| {
                    pieces.get(ordinal).copied().ok_or_else(|| {
                        StoreError::Invalid("a merged block has no stored piece".into())
                    })
                })
                .collect::<Result<_>>()?;
            Ok(StoredReading {
                message,
                selection,
                pieces,
            })
        })
        .transpose()
}

/// An event as a stored piece holds it: its fields, then its data.
pub(crate) fn read_event(
    conn: &Connection,
    reader: &mut crate::content::Reader,
    piece: crate::content::Piece,
) -> Result<Event> {
    let mut event: Event = reader.json(conn, piece.body())?;
    if let Some(data) = piece.data() {
        event.data = reader.json(conn, data)?;
    }
    Ok(event)
}

/// One reading of a shown event, with its fields as shown.
pub(crate) struct EventReading {
    pub input: String,
    pub format: String,
    pub chain: i64,
    pub start: Option<u64>,
    pub end: Option<u64>,
    pub imported: String,
    pub hash: Vec<u8>,
    pub fields: Vec<u8>,
}

/// The readings a shown event shows: those naming its ID, and those of the
/// copies its session is shown for whose events take that ID there.
pub(crate) fn event_readings(conn: &Connection, event_pk: i64) -> Result<Vec<EventReading>> {
    let (id, session): (String, Option<String>) = conn
        .prepare_cached(
            "SELECT e.id,s.id FROM event e LEFT JOIN session s ON s.pk=e.session_pk WHERE e.pk=?1",
        )?
        .query_row([event_pk], |row| <(String, Option<String>)>::try_from(row))?;
    let mut ids = super::identity::Identities::new(conn);
    let mut stated = vec![(Some(id.clone()), None)];
    if let Some(session) = &session {
        for copy in ids.copies(session)? {
            stated.push((None, Some(copy)));
        }
    }
    let mut found = Vec::new();
    let mut reader = crate::content::Reader::default();
    for (event_id, copy) in stated {
        let mut statement = conn.prepare_cached(
            "SELECT i.id,i.origin,r.chain,r.source_start,r.source_start+r.source_length,r.imported_sources,r.hash,
                 r.event_id,r.content_group,r.content_start,r.content_length,r.data_length
             FROM event_parse r JOIN input_ref i ON i.pk=r.input_pk
             WHERE (?1 IS NOT NULL AND r.event_id=?1) OR (?2 IS NOT NULL AND r.session_id=?2)",
        )?;
        let rows = statement
            .query_map(params![event_id, copy], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, Option<u64>>(3)?,
                    r.get::<_, Option<u64>>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, Vec<u8>>(6)?,
                    r.get::<_, String>(7)?,
                    crate::content::Piece::of(r, 8)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (input, format, chain, start, end, imported, hash, stated_id, piece) in rows {
            let mut event = read_event(conn, &mut reader, piece)?;
            let (moved, shown) = ids.event(&event, &stated_id)?;
            if shown != id {
                continue;
            }
            if let Some(moved) = moved {
                event = moved;
            }
            event.data = Json::Null;
            found.push(EventReading {
                input,
                format,
                chain,
                start,
                end,
                imported,
                hash,
                fields: serde_json::to_vec(&event)?,
            });
        }
    }
    Ok(found)
}
