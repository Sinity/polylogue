use super::*;
use polylogue_model::saved_queries::SavedRevision;
use query_language::identity::QuestionId;

#[test]
fn history_version_work_uses_immutable_ids_not_original_query_text() {
    let mut saved = SavedQuery {
        header: SavedQueryHeader {
            name: "focus".into(),
            selected_revision: "annotation:second@second".into(),
            query_id: Reference::Query {
                key: "01".repeat(32),
                parameters: vec![],
            },
        },
        seed: BlobHash([2; 32]),
        selection_question: QuestionId([1; 32]),
        query: SavedQueryRecord {
            name: "focus".into(),
            revisions: vec![
                SavedRevision {
                    revision: 1,
                    reference: "annotation:first@first".into(),
                    text: "sessions".into(),
                    hash: format!("query:{}", "01".repeat(32)),
                    saved_at: None,
                    author: "operator".into(),
                },
                SavedRevision {
                    revision: 2,
                    reference: "annotation:second@second".into(),
                    text: "sessions".into(),
                    hash: format!("query:{}", "01".repeat(32)),
                    saved_at: None,
                    author: "operator".into(),
                },
            ],
            saved_rows: vec![],
            watch: None,
            dependents: vec![],
        },
        dependents: vec![],
    };
    let before = saved.version().unwrap();
    // Version work follows the immutable revision IDs; the body text must not
    // enter the version.
    saved.query.revisions[0].text = format!("sessions{}", " ".repeat(128 * 1024));
    assert_eq!(saved.version().unwrap(), before);
    // A distinct accepted revision is the content witness.
    saved.query.revisions[0].reference = "annotation:third@third".into();
    assert_ne!(saved.version().unwrap(), before);
}
