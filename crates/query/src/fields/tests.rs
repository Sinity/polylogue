use super::*;

#[test]
fn every_non_special_field_has_sql() {
    let a = Aliases::plain();
    for f in FIELDS {
        if f.kind != Kind::Special {
            assert!(sql(f.unit, f.name, &a).is_some(), "{:?}.{}", f.unit, f.name);
        }
    }
}

#[test]
fn names_are_unique_per_unit() {
    let mut seen = std::collections::HashSet::new();
    for f in FIELDS {
        assert!(seen.insert((f.unit, f.name)), "{:?}.{}", f.unit, f.name);
    }
}
