use polylogue_model::{Author, Note, signal::Checkpoint};
use shared_signals::{ActiveRule, RuleVersion};

pub(super) fn author_name(author: &shared_signals::Author) -> &str {
    match author {
        shared_signals::Author::Operator => "operator",
        shared_signals::Author::Shipped => "shipped",
        shared_signals::Author::Agent(_) => "agent",
    }
}

pub(super) fn active(
    note: &Note,
    checkpoint: &Checkpoint,
    configuration: &shared_signals::Configuration,
) -> Result<ActiveRule, String> {
    let rule: shared_signals::Rule =
        serde_json::from_str(note.value.as_deref().ok_or("signal rule has no value")?)
            .map_err(|e| e.to_string())?;
    rule.validate(configuration).map_err(|e| e.to_string())?;
    let id = format!("authored:{}", note.first_note);
    let previous = checkpoint.rules.iter().find(|r| r.version.id == id);
    let revision = if checkpoint.rule_sources.get(&id) == Some(&note.id) {
        previous.map(|r| r.version.revision).unwrap_or(1)
    } else {
        previous
            .map_or(Some(1), |r| r.version.revision.checked_add(1))
            .ok_or("rule adoption ordinal overflow")?
    };
    let author = if note.author == Author::Operator {
        shared_signals::Author::Operator
    } else {
        shared_signals::Author::Agent(note.author.to_string())
    };
    Ok(ActiveRule {
        version: RuleVersion {
            product: "polylogue".into(),
            id,
            revision,
        },
        rule,
        author,
        via: "assertion".into(),
        live: true,
    })
}
