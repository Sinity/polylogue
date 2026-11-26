//! Calendar conversion at the millisecond index boundary. Logical boundaries
//! come from the caller's already resolved shared setting.
use crate::Result;
use jiff::{Timestamp, civil::Date};
use query_language::time::{DaySetting, label};

fn outside(problem: impl ToString) -> crate::StoreError {
    polylogue_model::Reason::OutsideCalendar {
        problem: problem.to_string(),
    }
    .into()
}

fn instant(ms: i64) -> Result<Timestamp> {
    Timestamp::from_millisecond(ms).map_err(outside)
}
/// Integer timestamps satisfying a precise boundary start at its ceiling.
pub fn boundary_ms(at: Timestamp) -> i64 {
    let ns = at.as_nanosecond();
    let ms = ns.div_euclid(1_000_000) + i128::from(ns.rem_euclid(1_000_000) != 0);
    i64::try_from(ms).expect("jiff timestamps fit the millisecond index")
}
pub fn date(day: &DaySetting, ms: i64) -> Result<Date> {
    day.logical_day(instant(ms)?).map_err(outside)
}
pub fn day_label(day: &DaySetting, ms: i64) -> Result<String> {
    Ok(label::day(date(day, ms)?))
}
pub fn week_label(day: &DaySetting, ms: i64) -> Result<String> {
    Ok(label::week(date(day, ms)?))
}
pub fn month_label(day: &DaySetting, ms: i64) -> Result<String> {
    Ok(label::month(date(day, ms)?))
}

pub(crate) fn weekday_label(day: &DaySetting, ms: i64) -> Result<String> {
    Ok(label::weekday(date(day, ms)?).into())
}
pub(crate) fn month_day_label(day: &DaySetting, ms: i64) -> Result<String> {
    Ok(label::month_day(date(day, ms)?))
}
pub(crate) fn hour_number(day: &DaySetting, ms: i64) -> Result<i64> {
    Ok(i64::from(instant(ms)?.to_zoned(day.zone().clone()).hour()))
}
pub(crate) fn year_number(day: &DaySetting, ms: i64) -> Result<i64> {
    Ok(i64::from(date(day, ms)?.year()))
}

#[cfg(test)]
mod tests;
