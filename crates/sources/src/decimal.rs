//! Decimal-to-integer scaling for money and elapsed-time fields.
use polylogue_model::RawJson as Value;

pub(crate) fn scaled_integer(value: &Value, scale: i32) -> Option<i64> {
    if !matches!(value, Value::Number(_)) {
        return None;
    }
    let raw = String::from_utf8(value.canonical()).ok()?;
    let (mantissa, exponent) = raw.split_once(['e', 'E']).unwrap_or((&raw, "0"));
    let exponent: i32 = exponent.parse().ok()?;
    let decimals = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let mut integer: i128 = mantissa.replace('.', "").parse().ok()?;
    let power = exponent
        .checked_add(scale)?
        .checked_sub(i32::try_from(decimals).ok()?)?;
    if power >= 0 {
        integer = integer.checked_mul(10i128.checked_pow(power.try_into().ok()?)?)?;
    } else {
        let divisor = 10i128.checked_pow(power.checked_neg()?.try_into().ok()?)?;
        if integer % divisor != 0 {
            return None;
        }
        integer /= divisor;
    }
    integer.try_into().ok()
}
