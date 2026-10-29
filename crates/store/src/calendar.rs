//! Calendar conversion at the millisecond index boundary. Logical boundaries
//! come from the caller's already resolved shared setting.
use crate::{Result, StoreError};
use jiff::{Timestamp, civil::Date};
use query_language::time::DaySetting;

fn invalid(error: impl ToString) -> StoreError {
    StoreError::Invalid(error.to_string())
}
pub fn instant(ms: i64) -> Result<Timestamp> {
    Timestamp::from_millisecond(ms).map_err(invalid)
}
/// Integer timestamps satisfying a precise boundary start at its ceiling.
pub fn boundary_ms(at: Timestamp) -> i64 {
    let ns = at.as_nanosecond();
    let ms = ns.div_euclid(1_000_000) + i128::from(ns.rem_euclid(1_000_000) != 0);
    i64::try_from(ms).expect("jiff timestamps fit the millisecond index")
}
pub fn date(day: &DaySetting, ms: i64) -> Result<Date> {
    day.logical_day(instant(ms)?).map_err(invalid)
}
pub fn start(day: &DaySetting, date: Date) -> Result<i64> {
    day.start_of(date).map(boundary_ms).map_err(invalid)
}
pub fn range(day: &DaySetting, date: Date) -> Result<(i64, i64)> {
    let range = day.day(date).map_err(invalid)?;
    Ok((boundary_ms(range.start), boundary_ms(range.end)))
}
pub fn week(day: &DaySetting, date: Date) -> Result<(i64, i64)> {
    let monday = date
        .checked_sub(jiff::Span::new().days(i64::from(date.weekday().to_monday_zero_offset())))
        .map_err(invalid)?;
    let next = monday
        .checked_add(jiff::Span::new().days(7))
        .map_err(invalid)?;
    Ok((start(day, monday)?, start(day, next)?))
}
pub fn month(day: &DaySetting, date: Date) -> Result<(i64, i64)> {
    let first = date.first_of_month();
    let next = first
        .checked_add(jiff::Span::new().months(1))
        .map_err(invalid)?;
    Ok((start(day, first)?, start(day, next)?))
}
pub fn day_label(day: &DaySetting, ms: i64) -> Result<String> {
    Ok(date(day, ms)?.to_string())
}
pub fn week_label(day: &DaySetting, ms: i64) -> Result<String> {
    let week = date(day, ms)?.iso_week_date();
    Ok(format!("{}-W{:02}", year_label(week.year()), week.week()))
}
pub fn month_label(day: &DaySetting, ms: i64) -> Result<String> {
    let date = date(day, ms)?;
    Ok(format!("{}-{:02}", year_label(date.year()), date.month()))
}

fn year_label(year: i16) -> String {
    if year < 0 {
        format!("-{:04}", year.unsigned_abs())
    } else {
        format!("{year:04}")
    }
}

#[cfg(test)]
mod tests;
