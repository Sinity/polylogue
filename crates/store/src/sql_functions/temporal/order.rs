//! Time-order predicates use exact source periods; only public distances use seconds.
use super::{Failure, evidence};
use query_language::ast::{OrderRelation, TimePoint};
use query_language::time::Time;
use rusqlite::{
    Connection, Error,
    functions::{Context, FunctionFlags},
};

#[derive(Clone, Copy)]
struct Period {
    start: i128,
    end: i128,
    coarse: bool,
}
impl From<Time> for Period {
    fn from(value: Time) -> Self {
        match value {
            Time::Instant { at } => Self {
                start: at.as_nanosecond(),
                end: at.as_nanosecond(),
                coarse: false,
            },
            Time::Span { start, end } => Self {
                start: start.as_nanosecond(),
                end: end.as_nanosecond(),
                coarse: true,
            },
        }
    }
}
impl Period {
    fn point(self, at: TimePoint) -> i128 {
        if at == TimePoint::End {
            self.end
        } else {
            self.start
        }
    }
    fn last(self) -> i128 {
        if self.coarse {
            self.end - 1
        } else {
            self.start
        }
    }
}

fn point(text: &str) -> Result<TimePoint, Failure> {
    match text {
        "when" => Ok(TimePoint::When),
        "start" => Ok(TimePoint::Start),
        "end" => Ok(TimePoint::End),
        "match.when" => Ok(TimePoint::MatchWhen),
        "match.start" => Ok(TimePoint::MatchStart),
        "match.end" => Ok(TimePoint::MatchEnd),
        _ => Err("unknown order time point".into()),
    }
}
fn periods(ctx: &Context<'_>) -> rusqlite::Result<Option<(Period, Period)>> {
    let outer: Option<Vec<u8>> = ctx.get(0)?;
    let outer_storage: String = ctx.get(1)?;
    let inner: Option<Vec<u8>> = ctx.get(2)?;
    let inner_storage: String = ctx.get(3)?;
    let outer = evidence(outer.as_deref(), &outer_storage).map_err(Error::UserFunctionError)?;
    let inner = evidence(inner.as_deref(), &inner_storage).map_err(Error::UserFunctionError)?;
    Ok(outer
        .period()
        .zip(inner.period())
        .map(|(a, b)| (a.into(), b.into())))
}

fn relates(
    a: Period,
    b: Period,
    relation: OrderRelation,
    within: Option<u128>,
    at: TimePoint,
) -> bool {
    let point = a.point(at);
    if let Some(window) = within {
        let greatest = if a.coarse || b.coarse {
            [
                (a.start, b.start),
                (a.start, b.last()),
                (a.last(), b.start),
                (a.last(), b.last()),
            ]
            .into_iter()
            .map(|(a, b)| (b - a).unsigned_abs())
            .max()
            .unwrap_or(0)
        } else {
            (b.start - point).unsigned_abs()
        };
        if greatest > window || window < ((a.end - a.start) as u128).max((b.end - b.start) as u128)
        {
            return false;
        }
    }
    match relation {
        OrderRelation::Near => true,
        OrderRelation::Before => {
            if a.coarse {
                a.end <= b.start
            } else {
                point < b.start
            }
        }
        OrderRelation::After => {
            if b.coarse {
                (if a.coarse { a.start } else { point }) >= b.end
            } else {
                (if a.coarse { a.start } else { point }) > b.start
            }
        }
        OrderRelation::During => {
            if a.coarse {
                b.coarse && a.start >= b.start && a.end <= b.end
            } else {
                point >= b.start && point < b.end
            }
        }
        OrderRelation::Containing => {
            if b.coarse {
                a.coarse && b.start >= a.start && b.end <= a.end
            } else if a.coarse {
                b.start >= a.start && b.start < a.end
            } else {
                b.start == a.start
            }
        }
    }
}

pub(super) fn register(conn: &Connection) -> rusqlite::Result<()> {
    let flags = FunctionFlags::SQLITE_UTF8.union(FunctionFlags::SQLITE_DETERMINISTIC);
    conn.create_scalar_function("pl_time_order", 7, flags, |ctx| {
        let relation: String = ctx.get(4)?;
        let relation = OrderRelation::parse(&relation)
            .ok_or_else(|| Error::UserFunctionError("unknown time order relation".into()))?;
        let within: Option<String> = ctx.get(5)?;
        let within = within
            .map(|value| {
                value
                    .parse::<u128>()
                    .map_err(|e| Error::UserFunctionError(Box::new(e)))
            })
            .transpose()?;
        let at: String = ctx.get(6)?;
        let at = point(&at).map_err(Error::UserFunctionError)?;
        // These sources retain occurrence evidence, but declare no state span.
        // Its uncertainty bounds must never stand in for state endpoints.
        if matches!(at, TimePoint::MatchStart | TimePoint::MatchEnd)
            || (at == TimePoint::MatchWhen
                && matches!(relation, OrderRelation::During | OrderRelation::Containing))
        {
            return Ok(None);
        }
        Ok(periods(ctx)?.map(|(a, b)| relates(a, b, relation, within, at)))
    })?;
    conn.create_scalar_function("pl_time_distance_key", 5, flags, |ctx| {
        let at: String = ctx.get(4)?;
        let at = point(&at).map_err(Error::UserFunctionError)?;
        if matches!(at, TimePoint::MatchStart | TimePoint::MatchEnd) {
            return Ok(None);
        }
        Ok(periods(ctx)?.map(|(a, b)| format!("{:039}", (b.start - a.point(at)).unsigned_abs())))
    })?;
    conn.create_scalar_function("pl_time_distance", 5, flags, |ctx| {
        let at: String = ctx.get(4)?;
        let at = point(&at).map_err(Error::UserFunctionError)?;
        if matches!(at, TimePoint::MatchStart | TimePoint::MatchEnd) {
            return Ok(None);
        }
        Ok(periods(ctx)?.map(|(a, b)| (b.start - a.point(at)) as f64 / 1_000_000_000.0))
    })?;
    Ok(())
}
