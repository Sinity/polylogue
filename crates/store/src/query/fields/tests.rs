use super::*;

#[test]
fn every_non_special_field_has_sql() {
    let a = Aliases::plain();
    for f in FIELDS {
        if f.kind != Kind::Special {
            assert!(f.expr(&a, 0).is_some(), "{:?}.{}", f.unit, f.name);
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

#[test]
fn exactly_the_special_fields_have_a_compile_rule() {
    for f in FIELDS {
        assert_eq!(
            f.kind == Kind::Special,
            f.compile.is_some(),
            "{:?}.{}",
            f.unit,
            f.name
        );
    }
}

/// The catalog declares every field; the store gives each declared field one
/// rule and has no rule for a field the catalog does not declare.
#[test]
fn the_store_has_a_rule_for_exactly_the_catalog_fields() {
    for declared in polylogue_query::fields::FIELDS {
        assert!(
            FIELDS
                .iter()
                .any(|f| f.unit == declared.unit && f.name == declared.name),
            "{:?}.{} has no store rule",
            declared.unit,
            declared.name
        );
    }
    for f in FIELDS {
        assert!(
            polylogue_query::fields::FIELDS
                .iter()
                .any(|declared| declared.unit == f.unit && declared.name == f.name),
            "{:?}.{} is not in the catalog",
            f.unit,
            f.name
        );
    }
}
