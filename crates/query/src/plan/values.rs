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
    query_language::depth::deeper(|| {
        match &node.source {
            Source::Unit {
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
        for (index, stage) in node.transforms.iter().enumerate() {
            let at = path.field("transforms").index(index).field("kind");
            match &stage.kind {
                TransformKind::Where { condition: c }
                | TransformKind::Scope {
                    condition: Some(c), ..
                } => condition(c, typing, at.field("condition"))?,
                TransformKind::Combine { query: q, .. }
                | TransformKind::Join { query: q, .. }
                | TransformKind::Diff { query: q } => query(q, typing, at.field("query"))?,
                TransformKind::Extension { arguments, .. } => {
                    extension(arguments, typing, at.field("arguments"))?
                }
                _ => {}
            }
        }
        Ok(())
    })
}

fn condition(c: &Condition, typing: &Typing, path: NodePath) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
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
            ConditionKind::Compare {
                left: Expression::Field { .. },
                right: Expression::Value { value },
                ..
            } => values(typing, &at.field("left"), value, position)?,
            ConditionKind::Compare {
                left: Expression::Value { value },
                right: Expression::Field { .. },
                ..
            } => values(typing, &at.field("right"), value, position)?,
            ConditionKind::Between {
                left: Expression::Field { .. },
                from,
                to,
                ..
            } => {
                values(typing, &at.field("left"), from, position)?;
                values(typing, &at.field("left"), to, position)?;
            }
            ConditionKind::In { query: q, .. } | ConditionKind::Order { query: q, .. } => {
                query(q, typing, at.field("query"))?
            }
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
    })
}

fn extension(
    arguments: &[ExtensionArgument],
    typing: &Typing,
    path: NodePath,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        for (index, argument) in arguments.iter().enumerate() {
            let at = path.index(index);
            match argument {
                ExtensionArgument::Condition(c) => condition(c, typing, at.field("condition"))?,
                ExtensionArgument::Query(q) => query(q, typing, at.field("query"))?,
                ExtensionArgument::Optional(arguments) => {
                    extension(arguments, typing, at.field("optional"))?
                }
                ExtensionArgument::List(arguments) => {
                    extension(arguments, typing, at.field("list"))?
                }
                _ => {}
            }
        }
        Ok(())
    })
}

/// Checks a value compared with a catalog field of a numeric kind. Other
/// kinds (times, words, references) and output columns keep their own rules.
fn values(
    typing: &Typing,
    operand: &NodePath,
    value: &Value,
    position: usize,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
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
    })
}
