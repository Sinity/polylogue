//! Shared time expressions and explicit clock.
pub use query_language::time::*;

/// A money literal (`2.90`, `$2.90`, `2.90usd`) in integer nano-USD, exact:
/// half ties round away from zero, and an amount outside the i64 range is
/// refused rather than saturated.
pub fn money_nusd(text: &str) -> Option<i64> {
    let text = text.trim().to_ascii_lowercase();
    let text = text
        .strip_suffix("usd")
        .unwrap_or(&text)
        .trim_start_matches('$');
    polylogue_model::money::nusd_from_usd(text)
}

#[cfg(test)]
mod tests;
