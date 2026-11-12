use super::*;

fn relationship_version(w: &TestArchive) -> polylogue_model::BlobHash {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader().unwrap().lineage_version().unwrap()
}

#[test]
fn exact_chain_replacement_keeps_the_relationship_revision_but_retraction_changes_it() {
    let w = test_archive();
    let mut original = batch(
        "relations",
        Route::Live,
        0,
        vec![row(
            0,
            text_message("m1", "child", Role::User, "hello", None),
        )],
        &["parent", "child"],
    );
    original.parse[0].output.links.push(polylogue_model::Link {
        parent: sid("parent"),
        child: sid("child"),
        kind: LinkKind::Continuation,
        copies_prefix: false,
        how: None,
        at: None,
        call_id: None,
        evidence: "synthetic explicit marker".into(),
    });
    let empty = relationship_version(&w);
    let stored = w
        .store_reading(original.clone(), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let before = relationship_version(&w);
    assert_ne!(before, empty);
    let mut replacement = original.clone();
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut replacement.parse[0]).unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(replacement.input, stored.chain, replacement.length, reading)
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    assert_eq!(
        relationship_version(&w),
        before,
        "exact reparse preserves links cursors"
    );
    let mut removed = original;
    removed.parse[0].output.links.clear();
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut removed.parse[0]).unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(removed.input, stored.chain, removed.length, reading)
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    assert_ne!(relationship_version(&w), before);
}

fn relation(parent: &str, child: &str) -> polylogue_model::Link {
    polylogue_model::Link {
        parent: sid(parent),
        child: sid(child),
        kind: LinkKind::Continuation,
        copies_prefix: false,
        how: None,
        at: None,
        call_id: None,
        evidence: "synthetic explicit marker".into(),
    }
}

fn child_relations(w: &TestArchive, parent: &str) -> Vec<crate::lineage::SessionLink> {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader().unwrap().child_links(&sid(parent)).unwrap()
}

#[test]
fn copied_prefix_enrichment_changes_visible_relations_and_the_cursor_version() {
    let w = test_archive();
    let mut original = batch("edge", Route::Live, 0, vec![], &["parent", "child"]);
    original.parse[0]
        .output
        .links
        .push(relation("parent", "child"));
    w.store_reading(original, Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let before = relationship_version(&w);
    assert!(!child_relations(&w, "parent")[0].copies_prefix);
    let mut copied = batch("physical-copy", Route::Live, 0, vec![], &[]);
    copied.parse[0]
        .output
        .prefix_copies
        .push(polylogue_model::PrefixCopy {
            parent: sid("parent"),
            child: sid("child"),
            explicit_members: false,
        });
    w.store_reading(copied, Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    assert!(child_relations(&w, "parent")[0].copies_prefix);
    assert_ne!(relationship_version(&w), before);
}

#[test]
fn alias_reidentification_changes_visible_relations_and_the_cursor_version() {
    let w = test_archive();
    let mut original = batch("copy-edge", Route::Live, 0, vec![], &["copy", "child"]);
    original.parse[0]
        .output
        .links
        .push(relation("copy", "child"));
    w.store_reading(original, Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let before = relationship_version(&w);
    assert_eq!(
        child_relations(&w, "copy")[0].parent,
        sid("copy").to_string()
    );
    let mut named = batch("native-name", Route::Live, 0, vec![], &["known"]);
    named.parse[0].output.sessions[0].aliases.push(sid("copy"));
    w.store_reading(named, Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    assert!(child_relations(&w, "copy").is_empty());
    let links = child_relations(&w, "known");
    assert_eq!(links.len(), 1);
    assert_eq!(
        (&links[0].parent, &links[0].child),
        (&sid("known").to_string(), &sid("child").to_string())
    );
    assert_ne!(relationship_version(&w), before);
}

#[test]
fn changing_only_canonical_relation_order_invalidates_the_cursor() {
    let w = test_archive();
    let mut partial = batch("partial", Route::Live, 0, vec![], &["root", "a"]);
    partial.parse[0].output.links.push(relation("root", "a"));
    let stored = w
        .store_reading(partial.clone(), Some(&SOURCE_RECORD))
        .unwrap();
    let mut other = batch("other", Route::Live, 0, vec![], &["root", "b"]);
    other.parse[0].output.links.push(relation("root", "b"));
    w.store_reading(other, Some(&SOURCE_RECORD)).unwrap();
    let mut precise = batch("precise", Route::Live, 0, vec![], &["root", "a"]);
    let mut anchored = relation("root", "a");
    anchored.at = Some(MessageId::parse("claude-code:anchor").unwrap());
    precise.parse[0].output.links.push(anchored);
    w.store_reading(precise, Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let before = relationship_version(&w);
    let links = child_relations(&w, "root");
    assert_eq!(
        links.iter().map(|l| l.child.as_str()).collect::<Vec<_>>(),
        ["claude-code:a", "claude-code:b"]
    );
    partial.parse[0].output.links.clear();
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut partial.parse[0]).unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(partial.input, stored.chain, partial.length, reading)
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let reordered = child_relations(&w, "root");
    assert_eq!(reordered, [links[1].clone(), links[0].clone()]);
    assert_ne!(relationship_version(&w), before);
}
