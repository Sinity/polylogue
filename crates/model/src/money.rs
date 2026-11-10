//! Money is integer nano-USD everywhere it is stored, summed or compared.
//! Dollars appear only at the edges: parsed exactly from decimal text, and
//! written as an exact decimal (JSON) or rounded once for a person (text).

use query_language::number::Decimal;

/// Nano-USD in one USD.
pub const NUSD_PER_USD: i64 = 1_000_000_000;

/// An amount of USD written as decimal text (`2.90`, `1e-3`, a JSON number's
/// exact spelling) in nano-USD, half ties rounded away from zero. `None` for
/// text that is not a finite decimal or an amount outside `i64` nano-USD.
pub fn nusd_from_usd(text: &str) -> Option<i64> {
    i64::try_from(Decimal::parse(text)?.scaled_i128(9)?).ok()
}

/// The exact decimal dollars of an amount: `1234500000` → `1.2345`,
/// `-5` → `-0.000000005`, `0` → `0`.
pub fn usd_text(nusd: i64) -> String {
    let whole = nusd.unsigned_abs() / NUSD_PER_USD.unsigned_abs();
    let fraction = format!("{:09}", nusd.unsigned_abs() % NUSD_PER_USD.unsigned_abs());
    let fraction = fraction.trim_end_matches('0');
    format!(
        "{}{whole}{}{fraction}",
        if nusd < 0 { "-" } else { "" },
        if fraction.is_empty() { "" } else { "." },
    )
}

/// The exact decimal dollars of an amount as a JSON number.
pub fn usd_number(nusd: i64) -> serde_json::Number {
    usd_text(nusd)
        .parse()
        .expect("decimal dollar text is a JSON number")
}

/// An amount for a person: whole cents from one cent up (`$12.35`), and
/// three significant digits below a cent, so a known small cost never reads
/// as `$0.00` (`$0.00123`). Rounding happens here, once, after summing.
pub fn usd_display(nusd: i64) -> String {
    let sign = if nusd < 0 { "-" } else { "" };
    let magnitude = nusd.unsigned_abs();
    const CENT: u64 = 10_000_000;
    if magnitude == 0 {
        return "$0".into();
    }
    if magnitude >= CENT {
        let cents = (magnitude + CENT / 2) / CENT;
        return format!("{sign}${}.{:02}", cents / 100, cents % 100);
    }
    // Below a cent: keep the three leading digits of the nano-USD figure.
    let digits = magnitude.ilog10() + 1;
    let drop = digits.saturating_sub(3);
    let unit = 10u64.pow(drop);
    let kept = (magnitude + unit / 2) / unit * unit;
    format!("{sign}${}", usd_text(kept as i64))
}

/// A known amount, written to JSON as `{"nusd": 1234500000, "usd": 1.2345}`:
/// the integer that every sum uses and the exact dollars beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Amount(pub i64);

impl Amount {
    pub fn nusd(self) -> i64 {
        self.0
    }
}

impl serde::Serialize for Amount {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut amount = serializer.serialize_struct("Amount", 2)?;
        amount.serialize_field("nusd", &self.0)?;
        amount.serialize_field("usd", &usd_number(self.0))?;
        amount.end()
    }
}

/// `numerator / denominator` rounded to the nearest integer, half ties away
/// from zero. The one rounding rule for integer money arithmetic.
pub fn rounded_quotient(numerator: i128, denominator: i128) -> Option<i128> {
    if denominator == 0 {
        return None;
    }
    let quotient = numerator.checked_div(denominator)?;
    let remainder = numerator.checked_rem(denominator)?;
    if remainder.unsigned_abs().checked_mul(2)? >= denominator.unsigned_abs() {
        let away = if (numerator < 0) == (denominator < 0) {
            1
        } else {
            -1
        };
        quotient.checked_add(away)
    } else {
        Some(quotient)
    }
}

#[cfg(test)]
mod tests;
