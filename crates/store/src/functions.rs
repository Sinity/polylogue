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
//! - `pl_day(ms, zone, start_minutes)`, `pl_week(…)`, `pl_month(…)`: the
//!   logical date, ISO week or month label of an instant.
//!
//! All are deterministic, so SQLite may use them in indexes and caches.

use polylogue_model::LogicalDay;
use rusqlite::Connection;
use rusqlite::functions::{Context, FunctionFlags};
use std::cell::RefCell;

const FLAGS: FunctionFlags = FunctionFlags::SQLITE_UTF8.union(FunctionFlags::SQLITE_DETERMINISTIC);

pub fn register(conn: &Connection) -> rusqlite::Result<()> {
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
    conn.create_aggregate_function("pl_percentile", 2, FLAGS, Percentile)?;
    conn.create_scalar_function("pl_day", 3, FLAGS, |ctx| label(ctx, LogicalDay::day_label))?;
    conn.create_scalar_function("pl_week", 3, FLAGS, |ctx| {
        label(ctx, LogicalDay::week_label)
    })?;
    conn.create_scalar_function("pl_month", 3, FLAGS, |ctx| {
        label(ctx, LogicalDay::month_label)
    })?;
    Ok(())
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
    zstd::stream::decode_all(packed.as_slice())
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
mod tests {
    use super::*;

    #[test]
    fn functions_read_compressed_text_and_logical_days() {
        let conn = Connection::open_in_memory().unwrap();
        register(&conn).unwrap();
        let packed = zstd::bulk::compress("Łódź café".as_bytes(), 3).unwrap();
        let (text, key, folded): (String, String, String) = conn
            .query_row(
                "SELECT pl_text(?1), pl_key(pl_text(?1)), pl_fold('Łódź')",
                [&packed],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(text, "Łódź café");
        assert_eq!(key, "lodz cafe");
        assert_eq!(folded, "Lódź");
        // 01:30 local in Warsaw on Oct 6 belongs to logical Oct 5.
        let ms = "2026-10-05T23:30:00Z"
            .parse::<jiff::Timestamp>()
            .unwrap()
            .as_millisecond();
        let (day, week, month): (String, String, String) = conn
            .query_row(
                "SELECT pl_day(?1, 'Europe/Warsaw', 240), pl_week(?1, 'Europe/Warsaw', 240),
                        pl_month(?1, 'Europe/Warsaw', 240)",
                [ms],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            (day.as_str(), week.as_str(), month.as_str()),
            ("2026-10-05", "2026-W41", "2026-10")
        );
        let (p90, p50, empty): (f64, f64, Option<f64>) = conn
            .query_row(
                "SELECT (SELECT pl_percentile(column1, 90) FROM (VALUES (1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(NULL))),
                        (SELECT pl_percentile(column1, 50) FROM (VALUES (3),(1),(2))),
                        (SELECT pl_percentile(column1, 50) FROM (VALUES (NULL)))",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!((p90, p50, empty), (9.0, 2.0, None));
        let none: Option<String> = conn
            .query_row("SELECT pl_day(NULL, 'UTC', 0)", [], |r| r.get(0))
            .unwrap();
        assert_eq!(none, None);
        assert!(
            conn.query_row("SELECT pl_day(0, 'Nowhere/Else', 0)", [], |r| r
                .get::<_, String>(0))
                .is_err()
        );
    }
}
