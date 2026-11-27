use super::*;
use crate::fields::ExpectedType as T;
use serde_json::json;

#[test]
fn unused_leaf_counted_below_declared_containers_and_unknown_arrays() {
    static FIELDS: Fields = Fields::new(
        "record",
        &[
            ("message", Rule::Read(T::Object)),
            ("message.content", Rule::Read(T::StringOrArray)),
            ("message.content[].text", Rule::Read(T::String)),
            ("saved", Rule::Kept("copied into row attrs")),
            ("ignored", Rule::Excluded("known but not indexed")),
        ],
    );
    assert_eq!(
        inspect(
            &json!({
                "message":{"content":[{"text":"a","future":{"x":1}},{"text":"b","future":{"x":2}}]},
                "saved":{"nested":[1,2]},"ignored":{"reason":true},"unknown":[[3],{"leaf":4}]
            }),
            &FIELDS
        )
        .unused,
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
fn kept_attributes_copy_wire_values_not_unknown_siblings() {
    static FIELDS: Fields = Fields::new(
        "record",
        &[("message.meta", Rule::Kept("provider metadata"))],
    );
    let source = json!({"message":{"meta":{"nested":[{"text":"given"}]},"other":{"x":1}}});
    assert_eq!(
        attributes(&source, &FIELDS),
        serde_json::from_value::<serde_json::Map<String, Json>>(
            json!({"message.meta":{"nested":[{"text":"given"}]}})
        )
        .unwrap()
    );
    assert_eq!(inspect(&source, &FIELDS).unused, ["message.other.x"]);
}

#[test]
fn kept_pointer_paths_decode_escaped_names() {
    static FIELDS: Fields = Fields::new("record", &[("/a~1b/~3", Rule::Kept("odd names"))]);
    let source = json!({"a/b":{"[]":1},"other":2});
    assert_eq!(
        attributes(&source, &FIELDS),
        serde_json::from_value::<serde_json::Map<String, Json>>(json!({"/a~1b/~3":1})).unwrap()
    );
}

#[test]
fn header_empties_listed_containers_and_keeps_everything_else() {
    assert_eq!(
        header(
            &json!({"a":{"x":1},"rows":[{"t":1}],"n":2,"scalar":3,"other":{"y":1}}),
            &["a", "rows", "scalar"]
        ),
        json!({"a":{},"rows":[],"n":2,"scalar":3,"other":{"y":1}})
    );
    assert_eq!(header(&json!([1]), &["a"]), json!([1]));
}

#[test]
fn empty_containers_and_null_are_values_at_their_own_paths() {
    static FIELDS: Fields = Fields::new("record", &[("read", Rule::Read(T::String))]);
    assert_eq!(
        inspect(&json!({"a":[],"b":{},"c":null,"read":[]}), &FIELDS).unused,
        ["a", "b", "c"]
    );
}

#[test]
fn wire_observation_counts_types_and_declarations_without_scalar_bodies() {
    static FIELDS: Fields = Fields::new(
        "test_record",
        &[
            ("type", Rule::Read(T::String)),
            ("items[].n", Rule::Read(T::Number)),
            ("map", Rule::Kept("complete map in attrs")),
        ],
    )
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
    static EMPTY: Fields = Fields::new("empty", &[]);
    let observed = inspect(
        &json!({"a.b":1,"a":{"b":2},"[]":3,"*":4,"/":5,"~":6,"list":[7]}),
        &EMPTY,
    );
    assert_eq!(
        observed
            .shapes
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "", "/a", "/a.b", "/a/b", "/list", "/list/[]", "/~0", "/~1", "/~2", "/~3"
        ]
    );
}

#[test]
fn literal_keys_do_not_impersonate_declared_nested_or_array_paths() {
    static FIELDS: Fields = Fields::new(
        "record",
        &[
            ("a.b", Rule::Read(T::Number)),
            ("list[].n", Rule::Read(T::Number)),
        ],
    );
    let source = json!({"a.b":1,"a":{"b":2},"list[].n":3,"list":[{"n":4}]});
    assert_eq!(inspect(&source, &FIELDS).unused, ["a.b", "list[].n"]);
    let observed = inspect(&source, &FIELDS);
    assert!(observed.shapes["/a.b"].expected.is_empty());
    assert_eq!(observed.shapes["/a/b"].expected, ["null", "number"]);
}

#[test]
fn declared_open_values_and_dynamic_members_differ_from_new_kept_object_fields() {
    static FIELDS: Fields = Fields::new(
        "record",
        &[
            ("saved", Rule::Kept("raw attributes")),
            ("ignored", Rule::Excluded("not indexed")),
            ("map", Rule::Kept("dynamic attributes")),
        ],
    )
    .shape(&["map"], &[]);
    const ANY: [&str; 6] = ["null", "boolean", "number", "string", "array", "object"];
    let got = inspect(
        &json!({"saved":{"new_field":1},"ignored":true,"map":{"id":{"n":2}}}),
        &FIELDS,
    );
    assert_eq!(got.shapes["/saved"].expected, ANY);
    assert!(got.shapes["/saved/new_field"].expected.is_empty());
    assert_eq!(got.shapes["/ignored"].expected, ANY);
    assert_eq!(got.shapes["/map/*/n"].expected, ANY);
}

#[test]
fn new_kind_is_counted_without_keeping_its_scalar_value() {
    static FIELDS: Fields = Fields::new("test_record", &[("type", Rule::Read(T::String))]).shape(
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
    static FIELDS: Fields = Fields::new(
        "record",
        &[
            ("mapping", Rule::Kept("native key map")),
            ("mapping.*.role", Rule::Read(T::String)),
            ("/literal/~2", Rule::Read(T::Number)),
        ],
    )
    .shape(&["mapping"], &[]);
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

#[test]
fn kept_declarations_with_array_paths_are_refused_or_resolved() {
    // A kept path names one attribute value; one under an array element or a
    // map wildcard names many and is refused.
    for path in [
        "content[].citations",
        "turns[]",
        "map.*.value",
        "/a/[]/b",
        "*",
    ] {
        assert_eq!(
            check(&[(path, Rule::Kept("many values"))]),
            Err(
                "a kept field path holds `[]` or `*`; declare it ReadWhole where the parser stores it"
            ),
            "{path}"
        );
        assert_eq!(check(&[(path, Rule::ReadWhole("many values"))]), Ok(()));
    }
    // The same values read whole by the element's parser are accepted, and
    // literal names that only contain the characters are one value each.
    for (path, rule) in [
        (
            "content[].citations",
            Rule::ReadWhole("in the block's attributes"),
        ),
        ("/a~3b/~2", Rule::Kept("escaped literal names")),
        ("a*b", Rule::Kept("a name holding an asterisk")),
    ] {
        assert_eq!(check(&[(path, rule)]), Ok(()), "{path}");
    }
    // A value read whole accounts for its descendants, and is not an attribute.
    static FIELDS: Fields = Fields::new(
        "record",
        &[
            ("content", Rule::Read(T::Array)),
            (
                "content[].input",
                Rule::ReadWhole("tool input in the call block"),
            ),
            ("meta", Rule::Kept("provider metadata")),
        ],
    );
    let source = json!({"content":[{"input":{"cmd":["ls"]}}],"meta":{"k":1}});
    assert!(inspect(&source, &FIELDS).unused.is_empty());
    assert_eq!(
        attributes(&source, &FIELDS),
        serde_json::from_value::<serde_json::Map<String, Json>>(json!({"meta":{"k":1}})).unwrap()
    );
}
