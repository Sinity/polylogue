use super::*;

#[test]
fn names_are_unique_and_valid_origins() {
    let mut names: Vec<_> = ORIGINS.iter().map(|o| o.name).collect();
    for n in &names {
        polylogue_model::Origin::new(n).unwrap();
    }
    names.sort();
    names.dedup();
    assert_eq!(names.len(), ORIGINS.len());
}

#[test]
fn imports_have_no_live_locations() {
    for o in ORIGINS.iter().filter(|o| o.route == Route::Import) {
        assert!(o.locations.is_empty(), "{}", o.name);
    }
}
