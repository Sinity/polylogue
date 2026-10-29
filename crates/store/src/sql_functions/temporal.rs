//! Exact temporal predicates over canonical native evidence, without zone I/O.
use crate::query::{fields::NativeTimeStorage, time::NativePredicate};
use polylogue_model::{Document, Event, Precision, TimeEvidence, Timestamp};
use query_language::semantics::Truth;
use query_language::time::{Time, TimeValue};
use rusqlite::{Connection, functions::FunctionFlags};

type Failure = Box<dyn std::error::Error + Send + Sync + 'static>;

fn evidence(bytes: &[u8], storage: &str) -> Result<(Option<Timestamp>, TimeEvidence), Failure> {
    let storage = NativeTimeStorage::from_code(storage).ok_or("unknown native time storage")?;
    let value = match storage {
        NativeTimeStorage::Tuple => serde_json::from_slice(bytes)?,
        NativeTimeStorage::Event => {
            let event: Event = serde_json::from_slice(&crate::encoding::unpack_bytes(bytes)?)?;
            (event.at, event.time_evidence())
        }
        NativeTimeStorage::Document => {
            let document: Document =
                serde_json::from_slice(&crate::encoding::unpack_bytes(bytes)?)?;
            (document.source_time, document.time_evidence())
        }
    };
    value.1.validate(value.0)?;
    Ok(value)
}

fn period(at: Option<Timestamp>, evidence: TimeEvidence) -> Option<Time> {
    let at: jiff::Timestamp = at?.into();
    if evidence.precision == Some(Precision::Instant) {
        return Some(Time::Instant { at });
    }
    let end = match evidence.until {
        Some(end) => end.into(),
        None => match evidence.precision {
            Some(Precision::Second | Precision::Minute) => {
                let seconds = if evidence.precision == Some(Precision::Second) {
                    1
                } else {
                    60
                };
                at.checked_add(jiff::SignedDuration::from_secs(seconds))
                    .ok()?
            }
            // A calendar grain needs the source's boundary or upper bound.
            // The operation owner's zone cannot establish native provenance.
            _ => return None,
        },
    };
    Some(Time::Span { start: at, end })
}

fn test(bytes: &[u8], storage: &str, predicate: &NativePredicate) -> Result<Option<bool>, Failure> {
    let (at, evidence) = evidence(bytes, storage)?;
    if matches!(predicate, NativePredicate::Known) {
        return Ok(Some(at.is_some()));
    }
    let Some(period) = period(at, evidence) else {
        return Ok(None);
    };
    match predicate {
        NativePredicate::Known => unreachable!("handled known predicate"),
        NativePredicate::Overlap { since, until } => Ok(Some(
            query_language::time::overlaps_bounds(period, *since, *until),
        )),
        NativePredicate::Compare { condition, invert } => {
            // An explicit source period supplies its bounds independently of
            // a calendar label. Whole-period comparisons are the shared rule.
            let value = match period {
                Time::Instant { at } => TimeValue::instant(at),
                period => TimeValue::new(period, Precision::Bounded, None)?,
            };
            let truth = value.compare(condition);
            let truth = if *invert { truth.not() } else { truth };
            Ok(match truth {
                Truth::True => Some(true),
                Truth::False => Some(false),
                Truth::Unknown => None,
            })
        }
    }
}

pub(super) fn register(conn: &Connection) -> rusqlite::Result<()> {
    let flags = FunctionFlags::SQLITE_UTF8.union(FunctionFlags::SQLITE_DETERMINISTIC);
    conn.create_scalar_function("pl_time_is_point", 2, flags, |ctx| {
        let bytes: Vec<u8> = ctx.get(0)?;
        let storage: String = ctx.get(1)?;
        let (at, evidence) =
            evidence(&bytes, &storage).map_err(rusqlite::Error::UserFunctionError)?;
        Ok(at.is_some() && evidence.precision == Some(Precision::Instant))
    })?;
    conn.create_scalar_function("pl_time_test", 3, flags, |ctx| {
        let bytes: Vec<u8> = ctx.get(0)?;
        let storage: String = ctx.get(1)?;
        let predicate: String = ctx.get(2)?;
        let predicate: NativePredicate = serde_json::from_str(&predicate)
            .map_err(|error| rusqlite::Error::UserFunctionError(Box::new(error)))?;
        test(&bytes, &storage, &predicate).map_err(rusqlite::Error::UserFunctionError)
    })?;
    Ok(())
}
