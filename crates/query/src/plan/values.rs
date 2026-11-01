//! Comparison values are checked against their field's declared kind when a
//! question is prepared, so a value no row can be compared with is refused
//! before the question has an identity, is saved, or runs.
use crate::{QueryError, ast::*, fields::Kind};
use query_language::{
    bind::BoundQuery,
    catalog::FieldKind,
    scope::{NodePath, Origin, Typing},
};

pub(super) fn check(bound: &BoundQuery) -> Result<(), QueryError> {
    query(bound.query(), bound.typing(), NodePath::default())
}

fn query(node: &Query, typing: &Typing, path: NodePath) -> Result<(), QueryError> {
    match &node.source {
        Source::Unit {
            condition: Some(c), ..
        }
        | Source::RecallLast {
            condition: Some(c), ..
        }
        | Source::Saved {
            condition: Some(c), ..
        }
        | Source::Result {
            condition: Some(c), ..
        } => condition(c, typing, path.field("source").field("condition"))?,
        _ => {}
    }
    for (index, attachment) in node.attach.iter().enumerate() {
        let at = path.field("attach").index(index);
        for (index, window) in attachment.windows.iter().enumerate() {
            if let Window::Where(c) = window {
                condition(c, typing, at.field("windows").index(index).field("where"))?;
            }
        }
    }
    for (index, stage) in node.stages.iter().enumerate() {
        let at = path.field("stages").index(index).field("kind");
        match &stage.kind {
            StageKind::Where { condition: c }
            | StageKind::Scope {
                condition: Some(c), ..
            } => condition(c, typing, at.field("condition"))?,
            StageKind::Combine { query: q, .. } | StageKind::Temporal { query: q, .. } => {
                query(q, typing, at.field("query"))?
            }
            StageKind::Extension { arguments, .. } => {
                extension(arguments, typing, at.field("arguments"))?
            }
            _ => {}
        }
    }
    Ok(())
}

fn condition(c: &Condition, typing: &Typing, path: NodePath) -> Result<(), QueryError> {
    let at = path.field("kind");
    let position = c.span.start;
    match &c.kind {
        ConditionKind::And { conditions } | ConditionKind::Or { conditions } => {
            for (index, c) in conditions.iter().enumerate() {
                condition(c, typing, at.field("conditions").index(index))?;
            }
        }
        ConditionKind::Not { condition: c } => condition(c, typing, at.field("condition"))?,
        ConditionKind::Match { value, .. } => {
            values(typing, &at.field("field"), value, position)?;
        }
        ConditionKind::Compare { operand, value, .. } => {
            if let Operand::Field { .. } = operand {
                values(typing, &at.field("operand"), value, position)?;
            }
        }
        ConditionKind::Between {
            operand, from, to, ..
        } => {
            if let Operand::Field { .. } = operand {
                values(typing, &at.field("operand"), from, position)?;
                values(typing, &at.field("operand"), to, position)?;
            }
        }
        ConditionKind::In { query: q, .. } => query(q, typing, at.field("query"))?,
        ConditionKind::Exists {
            condition: Some(c), ..
        }
        | ConditionKind::Count {
            condition: Some(c), ..
        } => condition(c, typing, at.field("condition"))?,
        ConditionKind::Sequence { steps, .. } => {
            for (index, c) in steps.iter().enumerate() {
                condition(c, typing, at.field("steps").index(index))?;
            }
        }
        ConditionKind::Extension { arguments, .. } => {
            extension(arguments, typing, at.field("arguments"))?
        }
        _ => {}
    }
    Ok(())
}

fn extension(
    arguments: &[ExtensionArgument],
    typing: &Typing,
    path: NodePath,
) -> Result<(), QueryError> {
    for (index, argument) in arguments.iter().enumerate() {
        let at = path.index(index);
        match argument {
            ExtensionArgument::Condition(c) => condition(c, typing, at.field("condition"))?,
            ExtensionArgument::Query(q) => query(q, typing, at.field("query"))?,
            ExtensionArgument::Optional(arguments) => {
                extension(arguments, typing, at.field("optional"))?
            }
            _ => {}
        }
    }
    Ok(())
}

/// Checks a value compared with a catalog field of a numeric kind. Other
/// kinds (times, words, references) and output columns keep their own rules.
fn values(
    typing: &Typing,
    operand: &NodePath,
    value: &Value,
    position: usize,
) -> Result<(), QueryError> {
    let Some(typed) = typing.operand(operand) else {
        return Ok(());
    };
    if !matches!(typed.result.origin, Origin::Catalog { .. }) {
        return Ok(());
    }
    let kind = match typed.result.field.kind {
        FieldKind::Money => Kind::Money,
        FieldKind::Duration => Kind::Duration,
        FieldKind::Number | FieldKind::Integer => Kind::Number,
        _ => return Ok(()),
    };
    let name = typed.result.field.name.as_str();
    match value {
        Value::Scalar { text, .. } => {
            kind.scalar(name, text)
                .map(|_| ())
                .map_err(|message| QueryError::BadValue {
                    field: name.to_owned(),
                    value: text.clone(),
                    position,
                    message,
                })
        }
        Value::Set { values: members } => {
            for member in members {
                values(typing, operand, member, position)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
