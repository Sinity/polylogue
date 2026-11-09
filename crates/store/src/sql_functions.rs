//! SQL functions registered on read and schema connections, so compiled queries can
//! read what the index stores compressed and group by the operator's
//! logical day without a second pass in Rust:
//!
//! - `pl_text(body)`: a `text.body` decompressed to UTF-8;
//! - `pl_block(body)`: a `block.body` decompressed to its JSON (text moved
//!   to `text` is not in it);
//! - `pl_fold(text)`: letters with a stroke folded, as the index does;
//! - `pl_key(text)`: [`polylogue_model::text::key`], for case- and
//!   accent-blind substring tests;
//! - `pl_random(text_key, decimal_seed)`: a seeded integer order value;
//! - `pl_day(ms)`, `pl_week(ms)`, `pl_month(ms)`: the
//!   logical date, ISO week or month label of an instant;
//! - `pl_live(live_facts, grew_at, now, part)`: one part of a session's live
//!   view at `now` ([`polylogue_model::live::view`]): `state`, `until`,
//!   `since`, `waiting_on`, `stall`, `process_alive` or `rank`;
//! - `pl_similarity(text_hash)`: the text's similarity to the query text of
//!   a `near:` ranking ([`with_similarity`]), NULL for a text with no
//!   vector or outside such a query.
//!
//! Calendar labels and similarities depend on the explicit operation scope.
//! The remaining functions are deterministic.

use query_language::time::{Clock, DaySetting};
use rusqlite::Connection;
use rusqlite::functions::{Context, FunctionFlags};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

const FLAGS: FunctionFlags = FunctionFlags::SQLITE_UTF8.union(FunctionFlags::SQLITE_DETERMINISTIC);

pub fn register(conn: &Connection) -> rusqlite::Result<()> {
    temporal::register(conn)?;
    conn.create_scalar_function("pl_source_alias", 2, FLAGS, |ctx| {
        let values: String = ctx.get(0)?;
        let fallback: String = ctx.get(1)?;
        let values: Vec<Option<String>> = serde_json::from_str(&values)
            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
        let targets = values
            .into_iter()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>();
        if targets.len() > 1 {
            return Err(rusqlite::Error::UserFunctionError(
                "native record alias has conflicting canonical messages".into(),
            ));
        }
        Ok(targets.into_iter().next().unwrap_or(fallback))
    })?;
    conn.create_scalar_function("pl_glob", 2, FLAGS, |ctx| {
        let text: Option<String> = ctx.get(0)?;
        let pattern: Option<String> = ctx.get(1)?;
        Ok(text
            .zip(pattern)
            .map(|(text, pattern)| query_language::glob::matches(&text, &pattern)))
    })?;
    aggregates::register(conn)?;
    conn.create_scalar_function("pl_live", 4, FLAGS, |ctx| {
        use polylogue_model::live::{LiveFacts, view};
        use rusqlite::types::Value;
        let facts: Option<LiveFacts> = ctx
            .get::<Option<String>>(0)?
            .map(|text| serde_json::from_str(&text))
            .transpose()
            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
        let grew_at: Option<i64> = ctx.get(1)?;
        let now: i64 = ctx.get(2)?;
        let part: String = ctx.get(3)?;
        let v = view(facts.as_ref(), grew_at, now);
        let text = |t: Option<&'static str>| t.map_or(Value::Null, |t| Value::Text(t.into()));
        let int = |n: Option<i64>| n.map_or(Value::Null, Value::Integer);
        Ok(match part.as_str() {
            "state" => text(Some(v.state.as_str())),
            "until" => int(v.until),
            "since" => int(v.since),
            "waiting_on" => text(v.waiting_on.map(|w| w.as_str())),
            "stall" => text(v.stall.map(|s| s.as_str())),
            "process_alive" => int(v.process_alive.map(i64::from)),
            "rank" => Value::Integer(i64::from(v.state.rank())),
            other => {
                return Err(rusqlite::Error::UserFunctionError(
                    format!("pl_live has no part {other:?}").into(),
                ));
            }
        })
    })?;
    conn.create_scalar_function("pl_assertion_ref", 2, FLAGS, |ctx| {
        let root: String = ctx.get(0)?;
        let id: String = ctx.get(1)?;
        Ok(
            polylogue_model::Reference::Assertion(polylogue_model::RevisionReference::exact(
                root, id,
            ))
            .to_string(),
        )
    })?;
    let versions = crate::derive::current_versions()
        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
    conn.create_scalar_function("pl_derived_current", 2, FLAGS, move |ctx| {
        let name: String = ctx.get(0)?;
        let version: String = ctx.get(1)?;
        Ok(versions.get(&name).is_some_and(|wanted| *wanted == version))
    })?;
    conn.create_scalar_function("pl_is_prompt", 2, FLAGS, |ctx| {
        let role: String = ctx.get(0)?;
        let kind: Option<String> = ctx.get(1)?;
        Ok(polylogue_model::Role::from_name(&role).is_some_and(|role| {
            polylogue_model::is_prompt(
                role,
                kind.as_deref()
                    .and_then(polylogue_model::PromptKind::from_name),
            )
        }))
    })?;
    conn.create_scalar_function("pl_text", 1, FLAGS, |ctx| {
        decompress(ctx, 0).map(|b| b.map(|b| String::from_utf8_lossy(&b).into_owned()))
    })?;
    conn.create_scalar_function("pl_block", 1, FLAGS, |ctx| {
        decompress(ctx, 0).map(|b| b.map(|b| String::from_utf8_lossy(&b).into_owned()))
    })?;
    conn.create_scalar_function("pl_fold", 1, FLAGS, |ctx| {
        Ok(ctx
            .get::<Option<String>>(0)?
            .map(|t| polylogue_model::text::fold(&t)))
    })?;
    conn.create_scalar_function("pl_key", 1, FLAGS, |ctx| {
        Ok(ctx
            .get::<Option<String>>(0)?
            .map(|t| polylogue_model::text::key(&t)))
    })?;
    conn.create_scalar_function("pl_random", 2, FLAGS, |ctx| {
        // Callers cast numeric row keys to TEXT. Hash opaque string IDs as
        // bytes so numeric-looking IDs keep their spelling and precision.
        let key: String = ctx.get(0)?;
        let seed: String = ctx.get(1)?;
        let seed: u64 = seed
            .parse()
            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
        let mut digest = blake3::Hasher::new();
        digest.update(b"polylogue-random-order-v1\0");
        digest.update(&seed.to_le_bytes());
        digest.update(key.as_bytes());
        let bytes: [u8; 8] = digest.finalize().as_bytes()[..8]
            .try_into()
            .expect("eight hash bytes");
        Ok(i64::from_le_bytes(bytes))
    })?;
    conn.create_aggregate_function("pl_percentile", 2, FLAGS, Percentile)?;
    conn.create_scalar_function("pl_similarity", 1, FunctionFlags::SQLITE_UTF8, |ctx| {
        let Some(hash) = ctx.get::<Option<Vec<u8>>>(0)? else {
            return Ok(None);
        };
        Ok(SIMILARITY.with(|s| s.borrow().as_ref().and_then(|s| s.of(&hash)).map(f64::from)))
    })?;
    Ok(())
}

thread_local! {
    /// The similarities `pl_similarity` answers from while a `near:` query
    /// runs on this thread.
    static SIMILARITY: RefCell<Option<std::sync::Arc<crate::vectors::Similarity>>> =
        const { RefCell::new(None) };
}

/// Runs `f` with `pl_similarity` answering from `similarity`; the previous
/// answer source comes back afterwards, also when `f` panics.
pub(crate) fn with_similarity<T>(
    similarity: std::sync::Arc<crate::vectors::Similarity>,
    f: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<std::sync::Arc<crate::vectors::Similarity>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            SIMILARITY.with(|s| *s.borrow_mut() = previous);
        }
    }
    let _restore = Restore(SIMILARITY.with(|s| s.borrow_mut().replace(similarity)));
    f()
}

/// `pl_percentile(x, p)`: the nearest-rank p-th percentile of the known
/// values (unknown ones are left out, as in `avg`); NULL when none is known.
struct Percentile;

impl rusqlite::functions::Aggregate<(Vec<f64>, f64), Option<f64>> for Percentile {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<(Vec<f64>, f64)> {
        Ok((vec![], 50.0))
    }

    fn step(&self, ctx: &mut Context<'_>, acc: &mut (Vec<f64>, f64)) -> rusqlite::Result<()> {
        if let Some(x) = ctx.get::<Option<f64>>(0)? {
            acc.0.push(x);
        }
        acc.1 = ctx.get::<f64>(1)?.clamp(0.0, 100.0);
        Ok(())
    }

    fn finalize(
        &self,
        _: &mut Context<'_>,
        acc: Option<(Vec<f64>, f64)>,
    ) -> rusqlite::Result<Option<f64>> {
        let Some((mut values, p)) = acc else {
            return Ok(None);
        };
        if values.is_empty() {
            return Ok(None);
        }
        values.sort_by(f64::total_cmp);
        let rank = ((p / 100.0) * values.len() as f64).ceil() as usize;
        Ok(Some(values[rank.clamp(1, values.len()) - 1]))
    }
}

fn decompress(ctx: &Context<'_>, i: usize) -> rusqlite::Result<Option<Vec<u8>>> {
    let Some(packed) = ctx.get::<Option<Vec<u8>>>(i)? else {
        return Ok(None);
    };
    crate::encoding::unpack_bytes(&packed)
        .map(Some)
        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))
}

/// One connection's calendar scope. Cached statements consult this scope at
/// execution, and nested reads restore the preceding resolved setting.
#[derive(Clone)]
pub(crate) struct CalendarContext(Arc<Mutex<Vec<DaySetting>>>);
impl CalendarContext {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
    pub(crate) fn with_clock<T>(&self, clock: &Clock, f: impl FnOnce() -> T) -> T {
        struct Restore(CalendarContext);
        impl Drop for Restore {
            fn drop(&mut self) {
                self.0.0.lock().expect("calendar scope lock").pop();
            }
        }
        self.0
            .lock()
            .expect("calendar scope lock")
            .push(clock.day.clone());
        let _restore = Restore(self.clone());
        f()
    }
}
pub(crate) fn register_calendar(
    conn: &Connection,
    context: &CalendarContext,
) -> rusqlite::Result<()> {
    for (name, label) in [
        (
            "pl_day",
            crate::calendar::day_label as fn(&DaySetting, i64) -> crate::Result<String>,
        ),
        ("pl_week", crate::calendar::week_label),
        ("pl_month", crate::calendar::month_label),
        ("pl_weekday", crate::calendar::weekday_label),
        ("pl_month_day", crate::calendar::month_day_label),
    ] {
        let context = context.clone();
        conn.create_scalar_function(name, 1, FunctionFlags::SQLITE_UTF8, move |ctx| {
            let Some(ms) = ctx.get::<Option<i64>>(0)? else {
                return Ok(None);
            };
            let day = context
                .0
                .lock()
                .expect("calendar scope lock")
                .last()
                .cloned()
                .ok_or_else(|| {
                    rusqlite::Error::UserFunctionError(
                        "calendar labels require an explicit owner clock".into(),
                    )
                })?;
            label(&day, ms)
                .map(Some)
                .map_err(|error| rusqlite::Error::UserFunctionError(Box::new(error)))
        })?;
    }
    for (name, number) in [
        (
            "pl_hour",
            crate::calendar::hour_number as fn(&DaySetting, i64) -> crate::Result<i64>,
        ),
        ("pl_year", crate::calendar::year_number),
    ] {
        let context = context.clone();
        conn.create_scalar_function(name, 1, FunctionFlags::SQLITE_UTF8, move |ctx| {
            let Some(ms) = ctx.get::<Option<i64>>(0)? else {
                return Ok(None);
            };
            let day = context
                .0
                .lock()
                .expect("calendar scope lock")
                .last()
                .cloned()
                .ok_or_else(|| {
                    rusqlite::Error::UserFunctionError(
                        "calendar values require an explicit owner clock".into(),
                    )
                })?;
            number(&day, ms)
                .map(Some)
                .map_err(|error| rusqlite::Error::UserFunctionError(Box::new(error)))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests;

mod aggregates;
mod temporal;
