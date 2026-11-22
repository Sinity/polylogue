//! Read one bounded population from the resolver's existing snapshot. Shared
//! input and version/layout facts are fetched once per key. Session kinds are
//! resolved by composition only when they can change its content winner.
use super::{FieldGroup, IdKind, MessageMember, Shown, named};
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::collections::{BTreeMap, BTreeSet};

pub(super) const BATCH: usize = 512;

pub(super) struct StoredMessage {
    pub members: Vec<MessageMember>,
    pub shown: Option<Shown>,
}

struct Source {
    version: i64,
    member: MessageMember,
}

struct Version {
    hash: Vec<u8>,
    layout: Option<Vec<u8>>,
}

pub(super) fn load(
    index: &Connection,
    pks: &[i64],
    layouts: &mut BTreeMap<Vec<u8>, Option<Vec<u8>>>,
) -> Result<BTreeMap<i64, StoredMessage>> {
    let pks: Vec<_> = pks
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let keys = serde_json::to_string(&pks)?;
    let mut sources = BTreeMap::<i64, Vec<Source>>::new();
    let mut versions = BTreeSet::new();
    let mut inputs = BTreeSet::new();
    let mut statement = index.prepare_cached(
        "SELECT s.message_pk,s.version_pk,s.input_pk,s.chain,s.source_start,s.source_end,
         s.session_pk,s.main,s.completion,s.route,s.seq,s.record_position,s.id_kind,s.order_time,s.at
         FROM json_each(?1) k CROSS JOIN message_source s ON s.message_pk=k.value
         ORDER BY s.message_pk,s.input_pk,s.chain",
    )?;
    let mut rows = statement.query([&keys])?;
    while let Some(row) = rows.next()? {
        let pk: i64 = row.get(0)?;
        let version = row.get(1)?;
        let member = MessageMember {
            input: row.get(2)?,
            input_id: String::new(),
            chain: row.get(3)?,
            start: row.get(4)?,
            end: row.get(5)?,
            session: row.get(6)?,
            main: row.get(7)?,
            explicit_session: None,
            completion: named(row.get(8)?)?,
            route: named(row.get(9)?)?,
            seq: row.get(10)?,
            position: row.get(11)?,
            content: row.get::<_, String>(12)? == IdKind::Content.as_str(),
            order_time: row.get(13)?,
            at: row.get(14)?,
            hash: Vec::new(),
        };
        versions.insert(version);
        inputs.insert(member.input);
        sources
            .entry(pk)
            .or_default()
            .push(Source { version, member });
    }
    drop(rows);
    let versions = version_rows(index, versions)?;
    let inputs = input_rows(index, inputs)?;
    let mut stored: BTreeMap<_, _> = pks
        .iter()
        .map(|&pk| {
            (
                pk,
                StoredMessage {
                    members: Vec::new(),
                    shown: None,
                },
            )
        })
        .collect();
    for (pk, sources) in sources {
        let members = &mut stored.get_mut(&pk).expect("requested message").members;
        for source in sources {
            let mut member = source.member;
            let version = versions.get(&source.version).ok_or_else(|| {
                StoreError::Invalid("a stored message source has no version".into())
            })?;
            member.hash = version.hash.clone();
            member.input_id = inputs
                .get(&member.input)
                .ok_or_else(|| StoreError::Invalid("a stored message source has no input".into()))?
                .clone();
            members.push(member);
        }
    }
    for version in versions.into_values() {
        layouts.insert(version.hash, version.layout);
    }
    let mut headers = index.prepare_cached(
        "SELECT m.pk,m.session_pk,m.input_pk,m.source_chain
         FROM json_each(?1) k CROSS JOIN message m ON m.pk=k.value ORDER BY m.pk",
    )?;
    let mut rows = headers.query([&keys])?;
    let mut visible = Vec::new();
    while let Some(row) = rows.next()? {
        let pk: i64 = row.get(0)?;
        let session = row.get(1)?;
        let input: i64 = row.get(2)?;
        let chain: Option<i64> = row.get(3)?;
        let stored = stored.get_mut(&pk).expect("requested message");
        if let Some(main) = stored
            .members
            .iter()
            .find(|member| member.input == input && Some(member.chain) == chain)
        {
            stored.shown = Some(Shown {
                hash: main.hash.clone(),
                session,
                layout: layouts.get(&main.hash).cloned().flatten(),
                routes: Vec::new(),
            });
            visible.push(pk);
        }
    }
    drop(rows);
    if !visible.is_empty() {
        let mut routes = index.prepare_cached(
            "SELECT r.message_pk,r.field_group,r.input_pk,r.chain
             FROM json_each(?1) k CROSS JOIN message_route r ON r.message_pk=k.value
             ORDER BY r.message_pk,r.field_group",
        )?;
        let mut rows = routes.query([serde_json::to_string(&visible)?])?;
        while let Some(row) = rows.next()? {
            let pk: i64 = row.get(0)?;
            let group: String = row.get(1)?;
            let input = row.get(2)?;
            let chain = row.get(3)?;
            let stored = stored.get_mut(&pk).expect("visible message");
            if let Some(source) = stored
                .members
                .iter()
                .find(|member| member.input == input && member.chain == chain)
            {
                let group = FieldGroup::from_name(&group)
                    .ok_or_else(|| StoreError::Invalid(format!("unknown field group {group}")))?;
                stored
                    .shown
                    .as_mut()
                    .expect("visible message")
                    .routes
                    .push((group, input, chain, source.hash.clone()));
            }
        }
    }
    for message in stored.values_mut() {
        if let Some(shown) = &mut message.shown {
            shown.routes.sort();
        }
    }
    Ok(stored)
}

fn version_rows(index: &Connection, ids: BTreeSet<i64>) -> Result<BTreeMap<i64, Version>> {
    let ids: Vec<_> = ids.into_iter().collect();
    let mut found = BTreeMap::new();
    let mut query = index.prepare_cached(
        "SELECT v.pk,v.version_hash,p.layout FROM json_each(?1) k
         CROSS JOIN message_version v ON v.pk=k.value
         LEFT JOIN message_block_selection p ON p.version_pk=v.pk ORDER BY v.pk",
    )?;
    for ids in ids.chunks(BATCH) {
        let mut rows = query.query([serde_json::to_string(ids)?])?;
        while let Some(row) = rows.next()? {
            found.insert(
                row.get(0)?,
                Version {
                    hash: row.get(1)?,
                    layout: row.get(2)?,
                },
            );
        }
    }
    Ok(found)
}

fn input_rows(index: &Connection, ids: BTreeSet<i64>) -> Result<BTreeMap<i64, String>> {
    let ids: Vec<_> = ids.into_iter().collect();
    let mut found = BTreeMap::new();
    let mut query = index.prepare_cached(
        "SELECT i.pk,i.id FROM json_each(?1) k CROSS JOIN input_ref i ON i.pk=k.value ORDER BY i.pk",
    )?;
    for ids in ids.chunks(BATCH) {
        let mut rows = query.query([serde_json::to_string(ids)?])?;
        while let Some(row) = rows.next()? {
            found.insert(row.get(0)?, row.get(1)?);
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests;
