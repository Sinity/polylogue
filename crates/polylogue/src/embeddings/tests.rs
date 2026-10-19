use super::*;

#[test]
fn long_texts_are_chunked_and_normalized() {
    let short = chunks("cafe\u{301}");
    assert_eq!(short, ["café"]);
    let long = "x".repeat(CHUNK_CHARS * 2 + 5);
    assert_eq!(chunks(&long).len(), 3);
    let v = unit(vec![3.0, 4.0]);
    assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
}
