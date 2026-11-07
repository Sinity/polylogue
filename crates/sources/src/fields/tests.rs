use super::*;
use crate::fields::ExpectedType as T;
use serde_json::json;

#[test]
fn unused_leaf_counted_below_declared_containers_and_unknown_arrays() {
    static FIELDS: Fields = Fields::new(&[
        ("message", Rule::Read(T::Object)),
        ("message.content", Rule::Read(T::StringOrArray)),
        ("message.content[].text", Rule::Read(T::String)),
        ("saved", Rule::Kept("copied into row attrs")),
        ("ignored", Rule::Excluded("known but not indexed")),
    ])
    .for_kind("record");
    assert_eq!(
        unaccounted(
            &json!({
                "message":{"content":[{"text":"a","future":{"x":1}},{"text":"b","future":{"x":2}}]},
                "saved":{"nested":[1,2]},"ignored":{"reason":true},"unknown":[[3],{"leaf":4}]
            }),
            &FIELDS
        ),
        [
            "ignored.reason",
            "message.content[].future.x",
            "message.content[].future.x",
            "unknown[].leaf",
            "unknown[][]"
        ]
    );
}

#[test]
fn retained_attributes_copy_wire_values_without_covering_unknown_siblings() {
    static FIELDS: Fields =
        Fields::new(&[("message.meta", Rule::Kept("provider metadata"))]).for_kind("record");
    let source = json!({"message":{"meta":{"nested":[{"text":"given"}]},"other":{"x":1}}});
    assert_eq!(
        attributes(&source, &FIELDS),
        serde_json::from_value::<serde_json::Map<String, Json>>(
            json!({"message.meta":{"nested":[{"text":"given"}]}})
        )
        .unwrap()
    );
    assert_eq!(unaccounted(&source, &FIELDS), ["message.other.x"]);
}

#[test]
fn empty_containers_and_null_are_values_at_their_own_paths() {
    static FIELDS: Fields = Fields::new(&[("read", Rule::Read(T::String))]).for_kind("record");
    assert_eq!(
        unaccounted(&json!({"a":[],"b":{},"c":null,"read":[]}), &FIELDS),
        ["a", "b", "c"]
    );
}

#[test]
fn declaration_lookup_is_shared_by_parsing_threads() {
    static FIELDS: Fields = Fields::new(&[("a[].x", Rule::Read(T::String))]).for_kind("record");
    std::thread::scope(|scope| {
        let readers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    assert_eq!(
                        unaccounted(&json!({"a":[{"x":1,"y":2}]}), &FIELDS),
                        ["a[].y"]
                    );
                    FIELDS.compiled() as *const _ as usize
                })
            })
            .collect();
        let addresses: Vec<_> = readers
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(addresses.iter().all(|value| *value == addresses[0]));
    });
}

#[test]
fn wire_observation_counts_types_and_declarations_without_scalar_bodies() {
    static FIELDS: Fields = Fields::new(&[
        ("type", Rule::Read(T::String)),
        ("items[].n", Rule::Read(T::Number)),
        ("map", Rule::Kept("complete map in attrs")),
    ])
    .shape(
        &["map"],
        &[KnownKinds {
            path: "type",
            values: &["message"],
        }],
    );
    let observed = inspect(
        &json!({"type":"message","items":[{"n":1},{"n":"distinctive-secret"}],"map":{"first":true,"second":false}}),
        &FIELDS,
    );
    assert_eq!(
        observed.shapes["/items/[]/n"].types,
        BTreeMap::from([("number".into(), 1), ("string".into(), 1)])
    );
    assert_eq!(observed.shapes["/items/[]/n"].expected, ["null", "number"]);
    assert_eq!(
        observed.shapes["/map/*"].types,
        BTreeMap::from([("boolean".into(), 2)])
    );
    assert!(observed.shapes["/map"].dynamic_map);
    assert_eq!(observed.shapes["/type"].known_kinds, ["message"]);
    let encoded = serde_json::to_string(&observed.shapes).unwrap();
    assert!(!encoded.contains("distinctive-secret"));
    assert!(!encoded.contains("first"));
    assert!(!encoded.contains("second"));
}

#[test]
fn observed_paths_distinguish_literal_keys_arrays_and_nested_fields() {
    static EMPTY: Fields = Fields::new(&[]).for_kind("empty");
    let observed = inspect(
        &json!({"a.b":1,"a":{"b":2},"[]":3,"*":4,"/":5,"~":6,"list":[7]}),
        &EMPTY,
    );
    for path in ["/a.b", "/a/b", "/~3", "/~2", "/~1", "/~0", "/list/[]"] {
        assert!(observed.shapes.contains_key(path), "{path}");
    }
}

#[test]
fn literal_keys_do_not_impersonate_declared_nested_or_array_paths() {
    static FIELDS: Fields = Fields::new(&[
        ("a.b", Rule::Read(T::Number)),
        ("list[].n", Rule::Read(T::Number)),
    ])
    .for_kind("record");
    let source = json!({"a.b":1,"a":{"b":2},"list[].n":3,"list":[{"n":4}]});
    assert_eq!(unaccounted(&source, &FIELDS), ["a.b", "list[].n"]);
    let observed = inspect(&source, &FIELDS);
    assert!(observed.shapes["/a.b"].expected.is_empty());
    assert_eq!(observed.shapes["/a/b"].expected, ["null", "number"]);
}

#[test]
fn declared_open_values_and_dynamic_members_differ_from_new_kept_object_fields() {
    static FIELDS: Fields = Fields::new(&[
        ("saved", Rule::Kept("raw attributes")),
        ("ignored", Rule::Excluded("not indexed")),
        ("map", Rule::Kept("dynamic attributes")),
    ])
    .shape(&["map"], &[])
    .for_kind("record");
    let got = inspect(
        &json!({"saved":{"new_field":1},"ignored":true,"map":{"id":{"n":2}}}),
        &FIELDS,
    );
    assert_eq!(got.shapes["/saved"].expected, ExpectedType::Any.types());
    assert!(got.shapes["/saved/new_field"].expected.is_empty());
    assert_eq!(got.shapes["/ignored"].expected, ExpectedType::Any.types());
    assert_eq!(got.shapes["/map/*/n"].expected, ExpectedType::Any.types());
}

#[test]
fn new_kind_is_counted_without_retaining_its_scalar_value() {
    static FIELDS: Fields = Fields::new(&[("type", Rule::Read(T::String))]).shape(
        &[],
        &[KnownKinds {
            path: "type",
            values: &["message"],
        }],
    );
    let got = inspect(&json!({"type":"distinctive-new-kind"}), &FIELDS);
    assert_eq!(got.shapes["/type"].unknown_kind_count, 1);
    assert!(
        !serde_json::to_string(&got.shapes)
            .unwrap()
            .contains("distinctive-new-kind")
    );
}

#[test]
fn typed_dynamic_members_override_open_map_expectations() {
    static FIELDS: Fields = Fields::new(&[
        ("mapping", Rule::Kept("native key map")),
        ("mapping.*.role", Rule::Read(T::String)),
        ("/literal/~2", Rule::Read(T::Number)),
    ])
    .shape(&["mapping"], &[])
    .for_kind("record");
    let got = inspect(
        &json!({"mapping":{"first":{"role":"assistant"},"second":{"role":1}},"literal":{"*":2}}),
        &FIELDS,
    );
    assert_eq!(got.shapes["/mapping/*/role"].expected, ["null", "string"]);
    assert_eq!(
        got.shapes["/mapping/*/role"].types,
        BTreeMap::from([("string".into(), 1), ("number".into(), 1)])
    );
    assert_eq!(got.shapes["/literal/~2"].expected, ["null", "number"]);
}
