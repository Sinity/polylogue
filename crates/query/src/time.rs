//! Shared time expressions and explicit clock.
pub use query_language::time::*;

/// Dollars to the product's integer nano-dollar storage unit.
pub fn money_nusd(text: &str) -> Option<i64> {
    let text = text.trim().to_ascii_lowercase();
    let text = text
        .strip_suffix("usd")
        .unwrap_or(&text)
        .trim_start_matches('$');
    let n: f64 = text.parse().ok()?;
    n.is_finite().then(|| (n * 1e9).round() as i64)
}
