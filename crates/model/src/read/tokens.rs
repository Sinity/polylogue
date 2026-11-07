//! One conservative presentation estimator, shared by pages, briefs and packs.
pub const TOKEN_ESTIMATOR: &str = "utf8-words-v1";
pub fn estimate_tokens(text: &str) -> usize {
    let words = text
        .split_whitespace()
        .map(|word| {
            let length = word.chars().count();
            if length > 16 { length.div_ceil(8) } else { 1 }
        })
        .fold(0usize, usize::saturating_add);
    text.len()
        .div_ceil(4)
        .max(words.saturating_mul(13).div_ceil(10))
}
