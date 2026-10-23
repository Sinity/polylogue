use super::{Scope, hidden_sessions};
use crate::{Reader, Result};
use serde::Serialize;

/// One kind of token: the sum over the messages that reported it, and how
/// many model messages did not.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TokenCount {
    pub tokens: u64,
    pub unknown_messages: u64,
    /// Unknown grouping prevents quantifying the missing request denominator.
    pub response_total_unknown: bool,
}

/// The four tokens plus cache writes, which some tools bill apart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TokensByKind {
    /// Input not served from a cache.
    pub input: TokenCount,
    pub cached_input: TokenCount,
    pub cache_write: TokenCount,
    /// Output, without reasoning tokens a tool reported apart.
    pub output: TokenCount,
    pub reasoning: TokenCount,
}

/// Money in USD: the known part, and how many messages it could not price.
/// A total with unpriced messages is partial and never ranked as complete.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Money {
    /// `None` when nothing in scope is priced.
    pub known_usd: Option<f64>,
    pub priced_messages: u64,
    pub unpriced_messages: u64,
    pub response_total_unknown: bool,
}

impl Money {
    pub fn partial(&self) -> bool {
        (self.unpriced_messages > 0 || self.response_total_unknown) && self.priced_messages > 0
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ModelUsage {
    /// `None` for messages that name no model.
    pub model: Option<String>,
    pub messages: u64,
    pub tokens: TokensByKind,
    pub api_cost: Money,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct UsageReport {
    /// Physical records, with known response rows and independent observations apart.
    pub grain: &'static str,
    /// Bounds select source record times, not established API spend times.
    pub time_basis: &'static str,
    pub sessions: u64,
    /// Response total, unknown when source items lack response boundaries.
    pub messages: Option<u64>,
    /// Known response rows; this subset is not the total when messages is None.
    pub known_response_messages: u64,
    pub unknown_boundary_items: u64,
    /// Stored assistant contributions written by a model, not API requests.
    pub recorded_model_items: u64,
    pub recorded_usage: RecordedUsage,
    pub tokens: TokensByKind,
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

pub(super) const RESPONSE_ROWS: &str = "m.role = 'assistant' AND m.written_by = 'model'
 AND coalesce(json_extract(m.attrs_json, '$.production.response_boundary_unknown'), 0) = 0
 AND NOT EXISTS(SELECT 1 FROM current_event e WHERE e.session_pk=m.session_pk AND e.kind='response_boundaries_unknown')";

const TOKEN_SUMS: &str = "
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
    tokens: TokensByKind,
    api: Money,
    reported: Money,
}

fn row(r: &rusqlite::Row<'_>, at: usize) -> rusqlite::Result<Row> {
    let u = |i: usize| -> rusqlite::Result<u64> {
        Ok(r.get::<_, Option<i64>>(at + i)?.unwrap_or(0) as u64)
    };
    let token_count = |i: usize| -> rusqlite::Result<TokenCount> {
        Ok(TokenCount {
            tokens: u(i)?,
            unknown_messages: u(i + 1)?,
            response_total_unknown: false,
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
            response_total_unknown: false,
        })
    };
    Ok(Row {
        messages: u(0)?,
        tokens: TokensByKind {
            input: token_count(2)?,
            cached_input: token_count(4)?,
            cache_write: token_count(6)?,
            output: token_count(8)?,
            reasoning: token_count(10)?,
        },
        api: money(12)?,
        reported: money(15)?,
    })
}

pub(super) fn report(reader: &Reader<'_>, scope: &Scope) -> Result<UsageReport> {
    let (cond, params) = scope.condition();
    let conn = reader.conn()?;
    let mut total = conn.query_row(
        &format!("SELECT {TOKEN_SUMS} FROM current_message m WHERE {RESPONSE_ROWS} AND {cond}"),
        rusqlite::params_from_iter(params.iter()),
        |r| row(r, 0),
    )?;
    let mut by_model = vec![];
    let mut stmt = conn.prepare(&format!(
        "SELECT m.model, {TOKEN_SUMS} FROM current_message m WHERE {RESPONSE_ROWS} AND {cond}
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
            tokens: x.tokens,
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
    let recorded_usage = recorded(reader, scope)?;
    let unknown_boundary_items = unknown_items(reader, &cond, &params)?;
    let unknown = unknown_boundary_items > 0
        || recorded_usage.observations > 0
        || unknown_scope(reader, scope)?;
    mark_unknown(&mut total.tokens, &mut total.api, unknown);
    total.reported.response_total_unknown = unknown;
    Ok(UsageReport {
        grain: "physical",
        time_basis: "source_record_time",
        sessions: conn.query_row(
            &format!("SELECT count(DISTINCT m.session_pk) FROM (SELECT session_pk,at FROM current_message UNION ALL SELECT session_pk,at FROM current_event) m WHERE {cond}"),
            rusqlite::params_from_iter(params.iter()),
            |r| r.get::<_, i64>(0).map(|n| n as u64),
        )?,
        messages: (!unknown).then_some(total.messages),
        known_response_messages: total.messages,
        unknown_boundary_items,
        recorded_model_items: total.messages + unknown_boundary_items,
        recorded_usage,
        tokens: total.tokens,
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
    /// Response total, unknown when source items lack response boundaries.
    pub messages: Option<u64>,
    pub known_response_messages: u64,
    /// None at logical grain: unassociated physical observations cannot be
    /// assigned to inherited response history.
    pub recorded_usage: Option<RecordedUsage>,
    pub tokens: TokensByKind,
    pub api_cost: Money,
}

/// Usage of each session in the scope (by session key), at both grains.
pub(super) fn per_session(reader: &Reader<'_>, scope: &Scope) -> Result<Vec<SessionUsage>> {
    let (cond, params) = scope.condition();
    let conn = reader.conn()?;
    let mut stmt = conn.prepare(&format!(
        "SELECT DISTINCT s.pk, s.id FROM (SELECT session_pk,at FROM current_message UNION ALL SELECT session_pk,at FROM current_event) m JOIN session s ON s.pk = m.session_pk
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
        let mut physical = totals(reader, &format!("m.session_pk = {pk}"))?;
        let own_scope = Scope {
            sessions: Some(vec![pk]),
            ..Scope::default()
        };
        let observations = recorded(reader, &own_scope)?;
        if observations.observations > 0 || unknown_scope(reader, &own_scope)? {
            physical.messages = None;
            mark_unknown(&mut physical.tokens, &mut physical.api_cost, true);
        }
        physical.recorded_usage = Some(observations);
        let keys = reader.logical_message_keys(&sid)?;
        let list = keys
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut logical = totals(reader, &format!("m.pk IN ({list})"))?;
        if physical.messages.is_none() {
            logical.messages = None;
            mark_unknown(&mut logical.tokens, &mut logical.api_cost, true);
        }
        out.push(SessionUsage {
            session: id,
            physical,
            logical,
        });
    }
    Ok(out)
}

/// TokensByKind and cost over the model messages a condition on `m` selects. The
/// condition holds only integer keys the store read itself.
fn totals(reader: &Reader<'_>, cond: &str) -> Result<Totals> {
    let mut x = reader.conn()?.query_row(
        &format!("SELECT {TOKEN_SUMS} FROM current_message m WHERE {RESPONSE_ROWS} AND {cond}"),
        [],
        |r| row(r, 0),
    )?;
    let unknown = unknown_items(reader, cond, &[])? > 0;
    mark_unknown(&mut x.tokens, &mut x.api, unknown);
    Ok(Totals {
        messages: (!unknown).then_some(x.messages),
        known_response_messages: x.messages,
        recorded_usage: None,
        tokens: x.tokens,
        api_cost: x.api,
    })
}

/// Independent readings have their own denominator. Known sums cover only
/// validated fresh measured reports; unconfirmed readings remain in Events.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RecordedUsage {
    pub observations: u64,
    pub measured_completion_observations: u64,
    pub unconfirmed_observations: u64,
    pub known_usage: polylogue_model::Usage,
    pub known_api_cost_usd: Option<f64>,
    pub unpriced_observations: u64,
    /// Observations whose provider spend/measurement time is absent.
    pub untimed_observations: u64,
}

pub(super) fn unknown_items(
    reader: &Reader<'_>,
    cond: &str,
    params: &[rusqlite::types::Value],
) -> Result<u64> {
    Ok(reader.conn()?.query_row(
        &format!(
            "SELECT count(*) FROM current_message m
        WHERE m.role='assistant' AND m.written_by='model' AND {cond} AND (
        json_extract(m.attrs_json,'$.production.response_boundary_unknown')=1 OR
        EXISTS(SELECT 1 FROM current_event e WHERE e.session_pk=m.session_pk AND e.kind='response_boundaries_unknown'))"
        ),
        rusqlite::params_from_iter(params.iter()),
        |row| row.get::<_, i64>(0).map(|n| n as u64),
    )?)
}

pub(super) fn unknown_scope(reader: &Reader<'_>, scope: &Scope) -> Result<bool> {
    // Missing boundary evidence is a property of the selected source scope;
    // a later record-time window cannot turn it into known boundaries.
    let scope = Scope {
        since: None,
        until: None,
        ..scope.clone()
    };
    let (cond, params) = scope.condition();
    Ok(reader.conn()?.query_row(
        &format!(
            "SELECT EXISTS(SELECT 1 FROM current_event m
        WHERE m.kind='response_boundaries_unknown' AND {cond})"
        ),
        rusqlite::params_from_iter(params.iter()),
        |r| r.get(0),
    )?)
}

pub(super) fn recorded(reader: &Reader<'_>, scope: &Scope) -> Result<RecordedUsage> {
    recorded_with(reader, scope, |_, _, _| Ok(()))
}

pub(super) fn recorded_with(
    reader: &Reader<'_>,
    scope: &Scope,
    mut visit: impl FnMut(Option<i64>, &polylogue_model::UsageObservation, Option<i64>) -> Result<()>,
) -> Result<RecordedUsage> {
    let mut out = RecordedUsage::default();
    super::observations::visit::<polylogue_model::UsageObservation>(
        reader,
        scope,
        polylogue_model::observations::USAGE_OBSERVATION,
        |_, at, value| {
            out.observations += 1;
            if value.observed_at.is_none() {
                out.untimed_observations += 1;
            }
            if value.completion_observed != Some(true) {
                out.unconfirmed_observations += 1;
                return Ok(());
            }
            out.measured_completion_observations += 1;
            let add = |sum: &mut Option<u64>, next: Option<u64>| -> Result<()> {
                if let Some(next) = next {
                    *sum = Some(sum.unwrap_or(0).checked_add(next).ok_or_else(|| {
                        crate::StoreError::Invalid("recorded usage sum exceeds u64".into())
                    })?);
                }
                Ok(())
            };
            add(&mut out.known_usage.input, value.usage.input)?;
            add(&mut out.known_usage.cached_input, value.usage.cached_input)?;
            add(&mut out.known_usage.cache_write, value.usage.cache_write)?;
            add(&mut out.known_usage.output, value.usage.output)?;
            add(&mut out.known_usage.reasoning, value.usage.reasoning)?;
            let cost =
                polylogue_model::Catalog::builtin().cost_nusd(value.model.as_deref(), &value.usage);
            if let Some(cost) = cost {
                *out.known_api_cost_usd.get_or_insert(0.0) += cost as f64 / 1e9;
            } else {
                out.unpriced_observations += 1;
            }
            visit(at, &value, cost)?;
            Ok(())
        },
    )?;
    Ok(out)
}

fn mark_unknown(tokens: &mut TokensByKind, money: &mut Money, unknown: bool) {
    for lane in [
        &mut tokens.input,
        &mut tokens.cached_input,
        &mut tokens.cache_write,
        &mut tokens.output,
        &mut tokens.reasoning,
    ] {
        lane.response_total_unknown = unknown;
    }
    money.response_total_unknown = unknown;
}
