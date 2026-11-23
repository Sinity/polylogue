use super::*;

/// The catalog offers exactly the declared fields: each declaration under
/// its unit with its closed set, through `session.` on units in a session
/// and through each relation prefix, and no field without a declaration.
/// Each declared name looks up to its own identity.
#[test]
fn catalog_matches_declarations() {
    let catalog = crate::catalog::catalog(&[]);
    for &id in FieldId::ALL {
        let declared = id.declaration();
        assert_eq!(lookup(declared.unit, declared.name), Some(id), "{id:?}");
        let offered = catalog
            .field(declared.unit.name(), declared.name)
            .unwrap_or_else(|| panic!("{id:?} is not in the catalog"));
        let values: Vec<String> = declared
            .values
            .unwrap_or(&[])
            .iter()
            .map(|v| (*v).to_owned())
            .collect();
        assert_eq!(offered.values, values, "{id:?}");
    }
    for unit in Unit::all() {
        let offered = &catalog.unit(unit.name()).expect("unit in catalog").fields;
        for field in offered {
            let declared = |unit: Unit, name: &str| lookup(unit, name).is_some();
            let known = declared(unit, &field.name)
                || (has_session(unit)
                    && (field.name == "session" || declared(Unit::Sessions, &field.name)))
                || field.name.split_once('.').is_some_and(|(prefix, rest)| {
                    relation(unit, prefix).is_some_and(|target| declared(target, rest))
                });
            assert!(known, "{}.{} has no declaration", unit.name(), field.name);
        }
    }
}

/// A whole number beyond `i64` is refused rather than rounded to a float.
#[test]
fn numbers_beyond_i64_are_refused_not_rounded() {
    assert_eq!(
        Kind::Number.scalar("depth", "9223372036854775807").unwrap(),
        "9223372036854775807"
    );
    for text in [
        "9223372036854775808",
        "-9223372036854775809",
        "+99999999999999999999",
    ] {
        assert_eq!(
            Kind::Number.scalar("depth", text),
            Err("depth is out of range for a whole number".to_string()),
            "{text}"
        );
    }
    assert_eq!(
        Kind::Number
            .scalar("depth", "-9223372036854775808")
            .unwrap(),
        "-9223372036854775808"
    );
    assert_eq!(Kind::Number.scalar("depth", "1.5").unwrap(), "1.5");
    assert_eq!(
        Kind::Number.scalar("depth", "abc"),
        Err("depth is a number".to_string())
    );
}
