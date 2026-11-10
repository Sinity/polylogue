//! Scalar query values and numeric aggregates preserve unknowns and validate physical values.
use super::{Connection, Context, FLAGS};
use rusqlite::{
    functions::Aggregate,
    types::{Value, ValueRef},
};
use std::collections::BTreeMap;

fn invalid(message: &str) -> rusqlite::Error {
    rusqlite::Error::UserFunctionError(message.to_owned().into())
}
fn finite(value: f64) -> rusqlite::Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("numeric expression requires a finite result"))
    }
}

pub(super) fn register(conn: &Connection) -> rusqlite::Result<()> {
    conn.create_scalar_function("pl_finite", 1, FLAGS, |ctx| {
        Ok(match ctx.get_raw(0) {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(value) => Value::Integer(value),
            ValueRef::Real(value) => Value::Real(finite(value)?),
            _ => return Err(invalid("arithmetic requires numbers")),
        })
    })?;
    // SQLite's floating sum can return NULL for opposing infinities reached
    // from finite inputs. The count of known inputs tells that failure apart
    // from a total that is unknown because it has no known inputs.
    conn.create_scalar_function("pl_finite_aggregate", 2, FLAGS, |ctx| {
        Ok(match ctx.get_raw(0) {
            ValueRef::Null if ctx.get::<i64>(1)? == 0 => Value::Null,
            ValueRef::Null => return Err(invalid("numeric aggregate overflow")),
            ValueRef::Integer(value) => Value::Integer(value),
            ValueRef::Real(value) => Value::Real(finite(value)?),
            _ => return Err(invalid("numeric aggregate requires numbers")),
        })
    })?;
    conn.create_scalar_function("pl_abs", 1, FLAGS, |ctx| {
        Ok(match ctx.get_raw(0) {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(value) => value
                .checked_abs()
                .map_or_else(|| Value::Real((value as f64).abs()), Value::Integer),
            ValueRef::Real(value) => Value::Real(finite(value)?.abs()),
            _ => return Err(invalid("abs requires a number")),
        })
    })?;
    conn.create_scalar_function("pl_query_text", 2, FLAGS, |ctx| {
        let Some(text) = ctx.get::<Option<String>>(0)? else {
            return Ok(None);
        };
        if !ctx.get::<bool>(1)? {
            return Ok(Some(text));
        }
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|_| invalid("text requires valid stored JSON"))?;
        match value {
            serde_json::Value::String(text) => Ok(Some(text)),
            serde_json::Value::Null => Ok(None),
            _ => Err(invalid("text takes a string value")),
        }
    })?;
    conn.create_scalar_function("pl_query_word", 2, FLAGS, |ctx| {
        let text: Option<String> = ctx.get(0)?;
        let needle: Option<String> = ctx.get(1)?;
        Ok(text.zip(needle).map(|(text, needle)| {
            let needle = needle.to_lowercase();
            text.split(|c: char| !c.is_alphanumeric())
                .any(|word| word.to_lowercase() == needle)
        }))
    })?;
    conn.create_scalar_function("pl_query_under", 2, FLAGS, |ctx| {
        let path: Option<String> = ctx.get(0)?;
        let root: Option<String> = ctx.get(1)?;
        Ok(path.zip(root).map(|(path, root)| {
            let absolute_root = root.starts_with('/');
            let root = root.trim_end_matches('/');
            if root.is_empty() && absolute_root {
                return path.starts_with('/');
            }
            path == root
                || path
                    .strip_prefix(root)
                    .is_some_and(|rest| rest.starts_with('/'))
        }))
    })?;
    // Money is integer nano-USD. An amount scaled by a fractional number
    // rounds half away from zero; one outside the i64 range is refused.
    conn.create_scalar_function("pl_money", 1, FLAGS, |ctx| {
        Ok(match ctx.get_raw(0) {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(value) => Value::Integer(value),
            ValueRef::Real(value) => {
                let rounded = finite(value)?.round();
                // i64::MIN and 2^63 are exact floats; the range is [-2^63, 2^63).
                let limit = -(i64::MIN as f64);
                if !(-limit..limit).contains(&rounded) {
                    return Err(invalid("an amount is outside the nano-USD range"));
                }
                Value::Integer(rounded as i64)
            }
            _ => return Err(invalid("arithmetic requires numbers")),
        })
    })?;
    // A kept money value is stored as exact
    // dollars (JSON text) and read back without passing through a float.
    conn.create_scalar_function("pl_kept_money", 1, FLAGS, |ctx| {
        let text: Option<String> = ctx.get(0)?;
        match text.as_deref() {
            None | Some("null") => Ok(None),
            Some(text) => polylogue_model::money::nusd_from_usd(text)
                .map(Some)
                .ok_or_else(|| invalid("a kept amount must be decimal dollars within range")),
        }
    })?;
    // An integer average: `numerator / denominator` rounded half away from
    // zero, unknown for an unknown operand or no rows.
    conn.create_scalar_function("pl_rounded_quotient", 2, FLAGS, |ctx| {
        let numerator: Option<i64> = ctx.get(0)?;
        let denominator: Option<i64> = ctx.get(1)?;
        Ok(numerator.zip(denominator).and_then(|(n, d)| {
            polylogue_model::money::rounded_quotient(i128::from(n), i128::from(d))
                .and_then(|q| i64::try_from(q).ok())
        }))
    })?;
    conn.create_scalar_function("pl_kept_time", 1, FLAGS, |ctx| {
        let value = match ctx.get_raw(0) {
            ValueRef::Null => return Ok(None),
            ValueRef::Integer(value) => return Ok(Some(value)),
            ValueRef::Text(_) => ctx.get::<String>(0)?,
            _ => {
                return Err(invalid(
                    "a kept time must be a timestamp or integer milliseconds",
                ));
            }
        };
        let timestamp = polylogue_model::Timestamp::parse_rfc3339(&value)
            .ok_or_else(|| invalid("a kept time must be a valid timestamp"))?;
        let milliseconds = timestamp.unix_ms();
        if polylogue_model::Timestamp::from_unix_ms(milliseconds) != Some(timestamp) {
            return Err(invalid(
                "this kept time has finer precision than scalar milliseconds",
            ));
        }
        Ok(Some(milliseconds))
    })?;
    conn.create_scalar_function("pl_kept_identity", 3, FLAGS, |ctx| {
        let keys: Option<String> = ctx.get(0)?;
        let keys = keys.ok_or_else(|| invalid("the kept row has no captured native identity"))?;
        let keys: Vec<serde_json::Value> = serde_json::from_str(&keys)
            .map_err(|_| invalid("a kept native identity must be a scalar tuple"))?;
        let index = usize::try_from(ctx.get::<i64>(1)?)
            .map_err(|_| invalid("a kept native identity index must be nonnegative"))?;
        let count = usize::try_from(ctx.get::<i64>(2)?)
            .map_err(|_| invalid("a kept native identity size must be nonnegative"))?;
        if keys.len() != count {
            return Err(invalid("the kept native identity has the wrong tuple size"));
        }
        match keys.get(index) {
            Some(serde_json::Value::String(value)) => Ok(Value::Text(value.clone())),
            Some(serde_json::Value::Bool(value)) => Ok(Value::Integer(i64::from(*value))),
            Some(serde_json::Value::Number(value)) => {
                if let Some(value) = value.as_i64() {
                    Ok(Value::Integer(value))
                } else if value.is_f64() {
                    Ok(Value::Real(finite(value.as_f64().expect("JSON float"))?))
                } else {
                    Err(invalid(
                        "the kept native identity integer is out of SQL range",
                    ))
                }
            }
            _ => Err(invalid(
                "a kept native identity must contain known scalar keys",
            )),
        }
    })?;
    conn.create_scalar_function("pl_lower", 1, FLAGS, |ctx| {
        Ok(ctx
            .get::<Option<String>>(0)?
            .map(|value| value.to_lowercase()))
    })?;
    conn.create_scalar_function("pl_text_length", 1, FLAGS, |ctx| {
        Ok(ctx
            .get::<Option<String>>(0)?
            .map(|value| value.chars().count() as i64))
    })?;
    conn.create_scalar_function("pl_json_length", 1, FLAGS, |ctx| {
        let Some(text) = ctx.get::<Option<String>>(0)? else {
            return Ok(None);
        };
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|_| invalid("len requires a valid stored list or text"))?;
        Ok(Some(match value {
            serde_json::Value::Array(values) => values.len() as i64,
            serde_json::Value::String(value) => value.chars().count() as i64,
            _ => return Err(invalid("len takes text or a list")),
        }))
    })?;
    conn.create_aggregate_function("pl_covered", 2, FLAGS, Covered)?;
    conn.create_aggregate_function("pl_correlation", 2, FLAGS, Correlation)?;
    conn.create_aggregate_function("pl_histogram", 2, FLAGS, Histogram)?;
    Ok(())
}

struct Covered;
impl Aggregate<Vec<(f64, f64)>, Option<f64>> for Covered {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<Vec<(f64, f64)>> {
        Ok(Vec::new())
    }
    fn step(&self, ctx: &mut Context<'_>, intervals: &mut Vec<(f64, f64)>) -> rusqlite::Result<()> {
        if let (Some(start), Some(end)) = (ctx.get::<Option<f64>>(0)?, ctx.get::<Option<f64>>(1)?) {
            intervals.push((finite(start)?, finite(end)?));
        }
        Ok(())
    }
    fn finalize(
        &self,
        _: &mut Context<'_>,
        intervals: Option<Vec<(f64, f64)>>,
    ) -> rusqlite::Result<Option<f64>> {
        let Some(mut intervals) = intervals.filter(|intervals| !intervals.is_empty()) else {
            return Ok(None);
        };
        intervals.retain(|(start, end)| end > start);
        intervals
            .sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)));
        let mut current = None::<(f64, f64)>;
        let mut total = 0.;
        for (start, end) in intervals {
            match current.as_mut() {
                Some((_, last)) if start <= *last => *last = last.max(end),
                _ => {
                    if let Some((start, end)) = current {
                        total = finite(total + (end - start))?;
                    }
                    current = Some((start, end));
                }
            }
        }
        if let Some((start, end)) = current {
            total = finite(total + (end - start))?;
        }
        Ok(Some(total))
    }
}

#[derive(Default)]
struct CorrelationState {
    n: u64,
    scale_x: f64,
    scale_y: f64,
    x: f64,
    y: f64,
    xx: f64,
    yy: f64,
    xy: f64,
}
struct Correlation;
impl Aggregate<CorrelationState, Option<f64>> for Correlation {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<CorrelationState> {
        Ok(CorrelationState::default())
    }
    fn step(&self, ctx: &mut Context<'_>, state: &mut CorrelationState) -> rusqlite::Result<()> {
        if let (Some(x), Some(y)) = (ctx.get::<Option<f64>>(0)?, ctx.get::<Option<f64>>(1)?) {
            let (x, y) = (finite(x)?, finite(y)?);
            // Scaling each axis keeps finite source values from overflowing
            // the variance products; Pearson correlation is scale invariant.
            if x.abs() > state.scale_x {
                let factor = state.scale_x / x.abs();
                state.x *= factor;
                state.xx *= factor * factor;
                state.xy *= factor;
                state.scale_x = x.abs();
            }
            if y.abs() > state.scale_y {
                let factor = state.scale_y / y.abs();
                state.y *= factor;
                state.yy *= factor * factor;
                state.xy *= factor;
                state.scale_y = y.abs();
            }
            let x = if state.scale_x == 0. {
                0.
            } else {
                x / state.scale_x
            };
            let y = if state.scale_y == 0. {
                0.
            } else {
                y / state.scale_y
            };
            state.n += 1;
            let dx = x - state.x;
            let dy = y - state.y;
            state.x += dx / state.n as f64;
            state.y += dy / state.n as f64;
            state.xx = finite(state.xx + dx * (x - state.x))?;
            state.yy = finite(state.yy + dy * (y - state.y))?;
            state.xy = finite(state.xy + dx * (y - state.y))?;
        }
        Ok(())
    }
    fn finalize(
        &self,
        _: &mut Context<'_>,
        state: Option<CorrelationState>,
    ) -> rusqlite::Result<Option<f64>> {
        let Some(state) = state.filter(|state| state.n >= 2 && state.xx > 0. && state.yy > 0.)
        else {
            return Ok(None);
        };
        Ok(Some(
            finite((state.xy / state.xx.sqrt()) / state.yy.sqrt())?.clamp(-1., 1.),
        ))
    }
}

struct Histogram;
type Bins = BTreeMap<String, (serde_json::Value, u64)>;
impl Aggregate<Bins, String> for Histogram {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<Bins> {
        Ok(BTreeMap::new())
    }
    fn step(&self, ctx: &mut Context<'_>, bins: &mut Bins) -> rusqlite::Result<()> {
        if matches!(ctx.get_raw(0), ValueRef::Null) {
            return Ok(());
        }
        let kind: String = ctx.get(1)?;
        let value = match kind.as_str() {
            "text" => serde_json::Value::String(ctx.get(0)?),
            "json" => serde_json::from_str(&ctx.get::<String>(0)?)
                .map_err(|_| invalid("histogram requires valid stored JSON"))?,
            "bool" => serde_json::Value::Bool(ctx.get::<i64>(0)? != 0),
            "time" => serde_json::Value::String(
                polylogue_model::Timestamp::from_unix_ms(ctx.get(0)?)
                    .ok_or_else(|| invalid("histogram time is out of range"))?
                    .to_string(),
            ),
            "money" => serde_json::Value::Number(polylogue_model::money::usd_number(
                ctx.get::<i64>(0)
                    .map_err(|_| invalid("histogram money must be integer nano-USD"))?,
            )),
            "number" | "duration" | "duration_seconds" => {
                let scale = match kind.as_str() {
                    "duration_seconds" => 1000.,
                    _ => 1.,
                };
                let value = finite(ctx.get::<f64>(0)?)? / scale;
                serde_json::json!(if value == 0. { 0. } else { value })
            }
            _ => return Err(invalid("histogram requires a known column kind")),
        };
        let key = serde_json::to_string(&value)
            .map_err(|_| invalid("histogram value cannot be encoded"))?;
        bins.entry(key).or_insert((value, 0)).1 += 1;
        Ok(())
    }
    fn finalize(&self, _: &mut Context<'_>, bins: Option<Bins>) -> rusqlite::Result<String> {
        let mut bins = bins.unwrap_or_default().into_values().collect::<Vec<_>>();
        bins.sort_by(|(left, _), (right, _)| match (left, right) {
            (serde_json::Value::Number(left), serde_json::Value::Number(right)) => {
                left.as_f64().unwrap().total_cmp(&right.as_f64().unwrap())
            }
            _ => left.to_string().cmp(&right.to_string()),
        });
        serde_json::to_string(
            &bins
                .into_iter()
                .map(|(value, count)| serde_json::json!({"value":value,"count":count}))
                .collect::<Vec<_>>(),
        )
        .map_err(|_| invalid("histogram cannot be encoded"))
    }
}
