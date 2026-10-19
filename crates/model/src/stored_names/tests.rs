use crate::{
    Block, BlockKind, LINK_HOW_NAMES, LINK_KIND_NAMES, LinkKind, PromptKind, Role, SessionKind,
    ToolAction, WorkEdgeKind, WorkNodeKind,
};

fn serde_name<T: serde::Serialize>(v: T) -> String {
    serde_json::to_value(v)
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned()
}

macro_rules! check {
    ($ty:ty) => {
        assert_eq!(<$ty>::ALL.len(), <$ty>::NAMES.len());
        for (v, name) in <$ty>::ALL.iter().zip(<$ty>::NAMES) {
            assert_eq!(v.as_str(), *name);
            assert_eq!(serde_name(*v), *name, "{} serde name", stringify!($ty));
            assert_eq!(<$ty>::from_name(name), Some(*v));
        }
    };
}

#[test]
fn stored_names_are_the_serde_names() {
    check!(SessionKind);
    check!(Role);
    check!(PromptKind);
    check!(WorkNodeKind);
    check!(WorkEdgeKind);
    check!(ToolAction);
}

#[test]
fn block_kind_names_are_the_serde_tags() {
    let blocks = [
        Block::Text { text: "t".into() },
        Block::Thinking {
            text: None,
            signature: None,
            redacted: false,
        },
        Block::ToolCall {
            call_id: None,
            tool: None,
            action: ToolAction::Other,
            input: serde_json::Value::Null,
            command: None,
            path: None,
        },
        Block::ToolResult {
            call_id: None,
            outcome: crate::ToolOutcome::Ok,
            output: None,
            data: None,
            attachments: vec![],
        },
        Block::Image(Default::default()),
        Block::File(Default::default()),
        Block::Other {
            type_name: "x".into(),
            data: serde_json::Value::Null,
        },
    ];
    let kinds: Vec<BlockKind> = blocks.iter().map(Block::kind).collect();
    assert_eq!(kinds, BlockKind::ALL);
    for b in &blocks {
        let tag = serde_json::to_value(b).unwrap()["type"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(tag, b.kind().as_str());
    }
}

#[test]
fn link_names_cover_every_kind_once() {
    assert_eq!(LINK_KIND_NAMES, ["continuation", "subagent"]);
    assert_eq!(
        LINK_HOW_NAMES,
        ["fork", "fresh", "after_compaction", "resume"]
    );
    for k in LinkKind::ALL {
        let (kind, how) = k.columns();
        assert_eq!(LinkKind::from_columns(kind, how), Some(*k));
    }
}
