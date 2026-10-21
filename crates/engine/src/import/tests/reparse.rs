use super::*;
use polylogue_sources::parse::{DocumentParser, Output, Parse};
use polylogue_testkit::{ClaudeHome, claude_user, delete_index};
use std::sync::LazyLock;

struct PreviousParser;

impl DocumentParser for PreviousParser {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut polylogue_sources::Records) {
        let parser = polylogue_sources::claude_code::open(Path::new("fixture.jsonl")).unwrap();
        let (steps, _) = parser.steps(None, bytes, 0).unwrap();
        for step in steps {
            out.sessions.extend(step.output.sessions);
            out.messages.extend(step.output.messages);
        }
        for session in &mut out.sessions {
            session.tool_version = Some("previous interpretation".into());
        }
        let mut obsolete = out.messages[0].clone();
        obsolete.message.id = polylogue_model::MessageId::parse("claude-code:obsolete").unwrap();
        obsolete.message.blocks = vec![polylogue_model::Block::Text {
            text: "previous extra message".into(),
        }];
        out.messages.push(obsolete);
        records.record(
            "synthetic previous record",
            &polylogue_model::Disposition::Materialized,
        );
    }
}

fn previous_open(_: &Path) -> Option<Parse> {
    Some(Parse::Document(Box::new(PreviousParser)))
}

static PREVIOUS: LazyLock<OriginSpec> = LazyLock::new(|| {
    let origin = ORIGINS.iter().find(|o| o.name == "claude-code").unwrap();
    OriginSpec {
        parser_version: 0,
        open: previous_open,
        ..*origin
    }
});

#[test]
fn a_parser_upgrade_from_retained_bytes_matches_a_fresh_index_rebuild() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("retained.jsonl");
    let raw = claude_user("u1", "s1", "current source message");
    std::fs::write(&path, &raw).unwrap();
    {
        let (mut e, _lock) = engine(&w, 64);
        let captured = Captured::read(&e.blobs, std::fs::File::open(&path).unwrap()).unwrap();
        e.import_captured(
            &path,
            &PREVIOUS,
            captured,
            None,
            &mut ImportReport::default(),
        )
        .unwrap();
        assert_eq!(
            w.texts("s1"),
            ["current source message", "previous extra message"]
        );
    }
    std::fs::remove_file(&path).unwrap();
    {
        let (mut e, _lock) = engine(&w, 64);
        let pass = e.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        assert_eq!(w.texts("s1"), ["current source message"]);
        assert_eq!(
            w.count("SELECT count(*) AS n FROM parsed WHERE parser_version=0"),
            0
        );
        let r = e.reads.reader().unwrap();
        assert_eq!(
            r.query("SELECT harness_version FROM session", &[]).unwrap()[0]["harness_version"],
            serde_json::Value::Null
        );
    }
    let current = w.texts("s1");
    let retained = w.count("SELECT count(*) AS n FROM a.extent");
    delete_index(&w.root);
    {
        let (mut e, _lock) = engine(&w, 64);
        let pass = e.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    assert_eq!(w.texts("s1"), current);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.extent"), retained);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 1);
}
