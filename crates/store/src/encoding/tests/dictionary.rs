//! `rows.dict` is generated only from the synthetic samples
//! (docs/testing.md, Row dictionary).

#[path = "../../../examples/row_dictionary/samples.rs"]
mod samples;

#[test]
fn row_dictionary_rebuilds_from_its_samples() {
    let rebuilt = samples::dictionary().unwrap();
    let checked_in = include_bytes!("../rows.dict");
    assert!(
        rebuilt == checked_in,
        "rows.dict differs from the dictionary built from its samples ({} bytes against {}); \
         regenerate it with `cargo run -p polylogue-store --example row_dictionary crates/store/src/encoding/rows.dict`",
        rebuilt.len(),
        checked_in.len()
    );
}
