use super::*;

#[test]
fn native_member_ids_are_anchored_and_do_not_use_original_basenames() {
    for (member, id) in [
        ("file-AbC.dat", "file-AbC"),
        ("images/file-AbC-original.pdf", "file-AbC"),
        ("file_0123abcd-uuid.png", "file_0123abcd"),
        ("7abc#file_0123abcd#p_0.jpg", "file_0123abcd"),
    ] {
        assert_eq!(member_id(member), Some(id));
    }
    for decoy in [
        "notes-file-AbC.dat",
        "conversation_asset_file_names.json",
        "file-.dat",
        "chart.pdf",
    ] {
        assert_eq!(member_id(decoy), None);
    }
}

#[test]
fn naming_and_library_sidecars_map_their_fields() {
    let mut entries = Vec::new();
    metadata(
        "conversation_asset_file_names.json",
        br#"{"file-AbC.dat":"r\u00e9sum\u00e9.pdf","file_0123":""}"#.as_slice(),
        |entry| {
            entries.push(entry);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(entries[0].source_id, "file-AbC");
    assert_eq!(entries[0].name.as_deref(), Some("résumé.pdf"));
    assert_eq!(entries[1].name.as_deref(), Some(""));
    entries.clear();
    metadata("library_files.json", br#"[{"file_id":"file_0123","file_name":"plot.png","mime_type":"image/png","file_size_bytes":0,"unknown_field":{"kept":"in raw input"}}]"#.as_slice(), |entry| { entries.push(entry); Ok(()) }).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].source_id, "file_0123");
    assert_eq!(entries[0].name.as_deref(), Some("plot.png"));
    assert_eq!(entries[0].size, Some(0));
    assert_eq!(entries[0].media_type.as_deref(), Some("image/png"));
}

#[test]
fn malformed_sidecars_cannot_be_accepted_from_a_valid_prefix() {
    let names = "conversation_asset_file_names.json";
    for (bytes, valid, category, expected, entries) in [
        (
            br#"{"file-A":"a"} trailing"#.as_slice(),
            br#"{"file-A":"a"}"#.as_slice(),
            serde_json::error::Category::Syntax,
            "trailing characters at line 1 column 16",
            vec![("file-A", "a")],
        ),
        (
            br#"{"file-A":"a","file-B":null}"#.as_slice(),
            br#"{"file-A":"a","file-B":"b"}"#.as_slice(),
            serde_json::error::Category::Data,
            "invalid type: null, expected a string at line 1 column 27",
            vec![("file-A", "a"), ("file-B", "b")],
        ),
    ] {
        let ScanError::Malformed(error) = metadata(names, bytes, |_| Ok(())).unwrap_err() else {
            panic!("expected malformed names sidecar");
        };
        assert_eq!(error.classify(), category);
        assert_eq!(error.to_string(), expected);
        let mut actual = Vec::new();
        metadata(names, valid, |entry| {
            actual.push((entry.source_id, entry.name.unwrap()));
            Ok(())
        })
        .unwrap();
        assert_eq!(
            actual,
            entries
                .into_iter()
                .map(|(id, name)| (id.to_owned(), name.to_owned()))
                .collect::<Vec<_>>()
        );
    }

    let ScanError::Callback(message) = metadata(
        "library_files.json",
        br#"[{"file_id":null}]"#.as_slice(),
        |_| Ok(()),
    )
    .unwrap_err() else {
        panic!("expected a refusal for a null file ID");
    };
    assert_eq!(message, "library entry has no string file_id");
    let mut entries = Vec::new();
    metadata(
        "library_files.json",
        br#"[{"file_id":"file-A"}]"#.as_slice(),
        |entry| {
            entries.push(entry);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        entries,
        [AssetMetadata {
            source_id: "file-A".into(),
            name: None,
            media_type: None,
            size: None
        }]
    );
}

#[test]
fn supplied_library_fields_with_unknown_types_do_not_silently_become_absent() {
    let name = "library_files.json";
    for (field, invalid, valid, expected, declared) in [
        (
            "file_name",
            "7",
            r#""plot.png""#,
            "library entry file_name must be a string or null",
            (Some("plot.png"), None, None),
        ),
        (
            "mime_type",
            "false",
            r#""image/png""#,
            "library entry mime_type must be a string or null",
            (None, Some("image/png"), None),
        ),
        (
            "file_size_bytes",
            "-1",
            "0",
            "library entry file_size_bytes must be an unsigned integer",
            (None, None, Some(0)),
        ),
        (
            "file_size_bytes",
            "1.5",
            "1",
            "library entry file_size_bytes must be an unsigned integer",
            (None, None, Some(1)),
        ),
    ] {
        let source = format!(r#"[{{"file_id":"file-Native","{field}":{invalid}}}]"#);
        let ScanError::Callback(message) =
            metadata(name, source.as_bytes(), |_| Ok(())).unwrap_err()
        else {
            panic!("expected a refusal for {source}");
        };
        assert_eq!(message, expected, "{source}");
        let source = format!(r#"[{{"file_id":"file-Native","{field}":{valid}}}]"#);
        let mut entries = Vec::new();
        metadata(name, source.as_bytes(), |entry| {
            entries.push(entry);
            Ok(())
        })
        .unwrap();
        let (name, media_type, size) = declared;
        assert_eq!(
            entries,
            [AssetMetadata {
                source_id: "file-Native".into(),
                name: name.map(str::to_owned),
                media_type: media_type.map(str::to_owned),
                size,
            }]
        );
    }
}

#[test]
fn members_with_empty_or_traversing_path_parts_have_no_identity() {
    for member in [
        "../file-AbC.dat",
        "a//file-AbC.dat",
        "images/./file-AbC.dat",
        "images\\file-AbC.dat",
        "/file-AbC.dat",
    ] {
        assert_eq!(member_id(member), None, "{member}");
    }
    for member in [
        "../library_files.json",
        "a//library_files.json",
        "dir\\library_files.json",
    ] {
        assert_eq!(sidecar_kind(member), None, "{member}");
    }
    assert_eq!(
        sidecar_kind("export/library_files.json"),
        Some("library_files.json")
    );
}

#[test]
fn absent_library_fields_are_unknown_and_empty_ids_are_rejected() {
    let mut entries = Vec::new();
    metadata(
        "library_files.json",
        br#"[{"file_id":"file-A"},{"file_id":"file-B","file_name":null,"mime_type":null,"file_size_bytes":null}]"#
            .as_slice(),
        |entry| {
            entries.push(entry);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(entries.len(), 2);
    for entry in &entries {
        assert_eq!(entry.name, None);
        assert_eq!(entry.media_type, None);
        assert_eq!(entry.size, None);
    }
    let ScanError::Callback(message) = metadata(
        "library_files.json",
        br#"[{"file_id":""},{"file_id":"file-B","file_name":null,"mime_type":null,"file_size_bytes":null}]"#
            .as_slice(),
        |_| Ok(()),
    )
    .unwrap_err() else {
        panic!("expected a refusal for an empty file ID");
    };
    assert_eq!(message, "library entry has no string file_id");
    assert_eq!(entries[0].source_id, "file-A");

    let names = "conversation_asset_file_names.json";
    for (bytes, valid, column) in [
        (
            br#"{".dat":"a"}"#.as_slice(),
            br#"{"file-A.dat":"a"}"#.as_slice(),
            11,
        ),
        (
            br#"{"":"a"}"#.as_slice(),
            br#"{"file-A":"a"}"#.as_slice(),
            7,
        ),
    ] {
        let ScanError::Malformed(error) = metadata(names, bytes, |_| Ok(())).unwrap_err() else {
            panic!("expected malformed names sidecar");
        };
        assert_eq!(error.classify(), serde_json::error::Category::Data);
        assert_eq!(
            error.to_string(),
            format!("asset name has an empty provider file ID at line 1 column {column}")
        );
        let mut entries = Vec::new();
        metadata(names, valid, |entry| {
            entries.push(entry);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            entries,
            [AssetMetadata {
                source_id: "file-A".into(),
                name: Some("a".into()),
                media_type: None,
                size: None
            }]
        );
    }
}
