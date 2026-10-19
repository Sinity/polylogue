use super::{Scope, hidden_sessions};
use crate::{Reader, Result};
use serde::Serialize;

/// One token lane: the sum over the messages that reported it, and how
/// many model messages did not.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Lane {
    pub tokens: u64,
    pub unknown_messages: u64,
}

/// The four lanes plus cache writes, which some tools bill apart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Lanes {
    /// Input not served from a cache.
    pub input: Lane,
    pub cached_input: Lane,
    pub cache_write: Lane,
    /// Output, without reasoning tokens a tool reported apart.
    pub output: Lane,
    pub reasoning: Lane,
}

/// Money in USD: the known part, and how many messages it could not price.
/// A total with unpriced messages is partial and never ranked as complete.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Money {
    /// `None` when nothing in scope is priced.
    pub known_usd: Option<f64>,
    pub priced_messages: u64,
    pub unpriced_messages: u64,
}

impl Money {
    pub fn partial(&self) -> bool {
        self.unpriced_messages > 0 && self.priced_messages > 0
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ModelUsage {
    /// `None` for messages that name no model.
    pub model: Option<String>,
    pub messages: u64,
    pub lanes: Lanes,
    pub api_cost: Money,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct UsageReport {
    /// Always `physical`: each model response once.
    pub grain: &'static str,
    pub sessions: u64,
    /// Model messages (the ones that can carry usage).
    pub messages: u64,
    pub lanes: Lanes,
    /// What the responses would have cost at API list prices, from the price
    /// catalog. Subscription credit is never added to it.
    pub api_cost: Money,
    /// The tool's own cost figures, where it gives them.
    pub reported_cost: Money,
    pub price_catalog: String,
    pub by_model: Vec<ModelUsage>,
    /// Models the catalog has no price for, with their message counts.
    pub unpriced_models: Vec<(String, u64)>,
    /// Hidden sessions included in these figures.
    pub hidden_sessions: u64,
}

const LANES: &str = "
    count(*), count(DISTINCT m.session_pk),
    coalesce(sum(m.input_tokens), 0), sum(m.input_tokens IS NULL),
    coalesce(sum(m.cache_read_tokens), 0), sum(m.cache_read_tokens IS NULL),
    coalesce(sum(m.cache_write_tokens), 0), sum(m.cache_write_tokens IS NULL),
    coalesce(sum(m.output_tokens), 0), sum(m.output_tokens IS NULL),
    coalesce(sum(m.reasoning_tokens), 0), sum(m.reasoning_tokens IS NULL),
    sum(m.cost_nusd), sum(m.cost_nusd IS NOT NULL), sum(m.cost_nusd IS NULL),
    sum(m.provider_cost_nusd), sum(m.provider_cost_nusd IS NOT NULL),
    sum(m.provider_cost_nusd IS NULL)";

struct Row {
    messages: u64,
    sessions: u64,
    lanes: Lanes,
    api: Money,
    reported: Money,
}

fn row(r: &rusqlite::Row<'_>, at: usize) -> rusqlite::Result<Row> {
    let u = |i: usize| -> rusqlite::Result<u64> {
        Ok(r.get::<_, Option<i64>>(at + i)?.unwrap_or(0) as u64)
    };
    let lane = |i: usize| -> rusqlite::Result<Lane> {
        Ok(Lane {
            tokens: u(i)?,
            unknown_messages: u(i + 1)?,
        })
    };
    let money = |i: usize| -> rusqlite::Result<Money> {
        let priced = u(i + 1)?;
        Ok(Money {
            known_usd: (priced > 0)
                .then(|| r.get::<_, Option<i64>>(at + i))
                .transpose()?
                .flatten()
                .map(|n| n as f64 / 1e9),
            priced_messages: priced,
            unpriced_messages: u(i + 2)?,
        })
    };
    Ok(Row {
        messages: u(0)?,
        sessions: u(1)?,
        lanes: Lanes {
            input: lane(2)?,
            cached_input: lane(4)?,
            cache_write: lane(6)?,
            output: lane(8)?,
            reasoning: lane(10)?,
        },
        api: money(12)?,
        reported: money(15)?,
    })
}

pub(super) fn report(reader: &Reader<'_>, scope: &Scope) -> Result<UsageReport> {
    let (cond, params) = scope.condition();
    let conn = reader.conn();
    let total = conn.query_row(
        &format!("SELECT {LANES} FROM message m WHERE m.role = 'assistant' AND {cond}"),
        rusqlite::params_from_iter(params.iter()),
        |r| row(r, 0),
    )?;
    let mut by_model = vec![];
    let mut stmt = conn.prepare(&format!(
        "SELECT m.model, {LANES} FROM message m WHERE m.role = 'assistant' AND {cond}
         GROUP BY m.model ORDER BY sum(m.cost_nusd) DESC NULLS LAST, count(*) DESC"
    ))?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |r| {
        Ok((r.get::<_, Option<String>>(0)?, row(r, 1)?))
    })?;
    for x in rows {
        let (model, x) = x?;
        by_model.push(ModelUsage {
            model,
            messages: x.messages,
            lanes: x.lanes,
            api_cost: x.api,
        });
    }
    let catalog = polylogue_model::Catalog::builtin();
    let unpriced_models = by_model
        .iter()
        .filter_map(|m| {
            let name = m.model.as_deref()?;
            catalog
                .price(name)
                .is_none()
                .then(|| (name.to_owned(), m.messages))
        })
        .collect();
    Ok(UsageReport {
        grain: "physical",
        sessions: total.sessions,
        messages: total.messages,
        lanes: total.lanes,
        api_cost: total.api,
        reported_cost: total.reported,
        price_catalog: catalog.version.clone(),
        by_model,
        unpriced_models,
        hidden_sessions: hidden_sessions(reader, scope)?,
    })
}

/// One session's usage at both grains (architecture 1, tokens at two
/// grains): `physical` counts the responses stored in the session, each
/// response once across the archive; `logical` counts what the session
/// shows, the responses it inherited through fork and resume included, so a
/// resumed session's logical figures repeat its parent's.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SessionUsage {
    pub session: String,
    pub physical: Totals,
    pub logical: Totals,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Totals {
    /// Model messages counted.
    pub messages: u64,
    pub lanes: Lanes,
    pub api_cost: Money,
}

/// Usage of each session in the scope (by session key), at both grains.
pub(super) fn per_session(reader: &Reader<'_>, scope: &Scope) -> Result<Vec<SessionUsage>> {
    let (cond, params) = scope.condition();
    let conn = reader.conn();
    let mut stmt = conn.prepare(&format!(
        "SELECT DISTINCT s.pk, s.id FROM message m JOIN session s ON s.pk = m.session_pk
         WHERE {cond} ORDER BY s.started_at, s.id"
    ))?;
    let sessions: Vec<(i64, String)> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |r| {
            Ok((r.get(0)?, r.get(1)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::with_capacity(sessions.len());
    for (pk, id) in sessions {
        let Ok(sid) = polylogue_model::SessionId::parse(&id) else {
            continue;
        };
        let physical = totals(reader, &format!("m.session_pk = {pk}"))?;
        let keys = reader.logical_message_keys(&sid)?;
        let list = keys
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let logical = totals(reader, &format!("m.pk IN ({list})"))?;
        out.push(SessionUsage {
            session: id,
            physical,
            logical,
        });
    }
    Ok(out)
}

/// Lanes and cost over the model messages a condition on `m` selects. The
/// condition holds only integer keys the store read itself.
fn totals(reader: &Reader<'_>, cond: &str) -> Result<Totals> {
    let x = reader.conn().query_row(
        &format!("SELECT {LANES} FROM message m WHERE m.role = 'assistant' AND {cond}"),
        [],
        |r| row(r, 0),
    )?;
    Ok(Totals {
        messages: x.messages,
        lanes: x.lanes,
        api_cost: x.api,
    })
}
