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
    for bytes in [
        br#"{"file-A":"a"} trailing"#.as_slice(),
        br#"{"file-A":"a","file-B":null}"#.as_slice(),
    ] {
        assert!(metadata("conversation_asset_file_names.json", bytes, |_| Ok(())).is_err());
    }
    assert!(
        metadata(
            "library_files.json",
            br#"[{"file_id":null}]"#.as_slice(),
            |_| Ok(())
        )
        .is_err()
    );
}

#[test]
fn supplied_library_fields_with_unknown_types_do_not_silently_become_absent() {
    for field in [
        r#""file_name":7"#,
        r#""mime_type":false"#,
        r#""file_size_bytes":-1"#,
        r#""file_size_bytes":1.5"#,
    ] {
        let source = format!(r#"[{{"file_id":"file-Native",{field}}}]"#);
        assert!(
            metadata("library_files.json", source.as_bytes(), |_| Ok(())).is_err(),
            "accepted {source}"
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
    assert!(
        metadata(
            "library_files.json",
            br#"[{"file_id":""}]"#.as_slice(),
            |_| Ok(())
        )
        .is_err()
    );
    for bytes in [br#"{".dat":"a"}"#.as_slice(), br#"{"":"a"}"#.as_slice()] {
        assert!(metadata("conversation_asset_file_names.json", bytes, |_| Ok(())).is_err());
    }
}
