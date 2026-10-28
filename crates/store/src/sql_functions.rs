//! SQL functions registered on read connections, so compiled queries can
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
//! - `pl_day(ms, zone, start_minutes)`, `pl_week(…)`, `pl_month(…)`: the
//!   logical date, ISO week or month label of an instant;
//! - `pl_similarity(text_hash)`: the text's similarity to the query text of
//!   a `near:` ranking ([`with_similarity`]), NULL for a text with no
//!   vector or outside such a query.
//!
//! All but `pl_similarity` are deterministic, so SQLite may use them in
//! indexes and caches.

use polylogue_model::LogicalDay;
use rusqlite::Connection;
use rusqlite::functions::{Context, FunctionFlags};
use std::cell::RefCell;

const FLAGS: FunctionFlags = FunctionFlags::SQLITE_UTF8.union(FunctionFlags::SQLITE_DETERMINISTIC);

pub fn register(conn: &Connection) -> rusqlite::Result<()> {
    let versions = crate::derive::current_versions()
        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
    conn.create_scalar_function("pl_derived_current", 2, FLAGS, move |ctx| {
        let name: String = ctx.get(0)?;
        let version: String = ctx.get(1)?;
        Ok(versions.get(&name).is_some_and(|wanted| *wanted == version))
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
    conn.create_scalar_function("pl_day", 3, FLAGS, |ctx| label(ctx, LogicalDay::day_label))?;
    conn.create_scalar_function("pl_week", 3, FLAGS, |ctx| {
        label(ctx, LogicalDay::week_label)
    })?;
    conn.create_scalar_function("pl_month", 3, FLAGS, |ctx| {
        label(ctx, LogicalDay::month_label)
    })?;
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

thread_local! {
    /// The last zone looked up: every row of one query names the same one.
    static ZONE: RefCell<Option<(String, u16, LogicalDay)>> = const { RefCell::new(None) };
}

fn label(ctx: &Context<'_>, f: fn(&LogicalDay, i64) -> String) -> rusqlite::Result<Option<String>> {
    let Some(ms) = ctx.get::<Option<i64>>(0)? else {
        return Ok(None);
    };
    let zone: String = ctx.get(1)?;
    let start: i64 = ctx.get(2)?;
    let start = u16::try_from(start.clamp(0, 1439)).unwrap_or(0);
    ZONE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let hit = matches!(&*cache, Some((z, s, _)) if *z == zone && *s == start);
        if !hit {
            let day = LogicalDay::named(&zone, start).ok_or_else(|| {
                rusqlite::Error::UserFunctionError(format!("unknown time zone {zone:?}").into())
            })?;
            *cache = Some((zone.clone(), start, day));
        }
        let (_, _, day) = cache.as_ref().expect("filled above");
        Ok(Some(f(day, ms)))
    })
}

#[cfg(test)]
mod tests;
