//! Comparison values are checked against their field's declared kind when a
//! question is prepared, so a value no row can be compared with is refused
//! before the question has an identity, is saved, or runs.
use crate::{QueryError, ast::*, fields::Kind};
use query_language::{
    catalog::FieldKind,
    scope::{NodePath, Origin, Typing},
};

pub(super) fn normalize(query: &mut Query, typing: &Typing) -> Result<(), QueryError> {
    visit(query, typing, NodePath::default(), true)
}
pub(super) fn check(query: &mut Query, typing: &Typing) -> Result<(), QueryError> {
    visit(query, typing, NodePath::default(), false)
}

fn visit(
    node: &mut Query,
    typing: &Typing,
    path: NodePath,
    normalize: bool,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        match &mut node.source {
            Source::Unit {
                condition: Some(c), ..
            }
            | Source::Saved {
                condition: Some(c), ..
            }
            | Source::Result {
                condition: Some(c), ..
            } => condition(
                c,
                typing,
                path.field("source").field("condition"),
                normalize,
            )?,
            _ => {}
        }
        for (index, attachment) in node.attach.iter_mut().enumerate() {
            let at = path.field("attach").index(index);
            for (index, window) in attachment.windows.iter_mut().enumerate() {
                if let Window::Where(c) = window {
                    condition(
                        c,
                        typing,
                        at.field("windows").index(index).field("where"),
                        normalize,
                    )?;
                }
            }
        }
        for (index, stage) in node.transforms.iter_mut().enumerate() {
            let at = path.field("transforms").index(index).field("kind");
            match &mut stage.kind {
                TransformKind::Where { condition: c }
                | TransformKind::Scope {
                    condition: Some(c), ..
                } => condition(c, typing, at.field("condition"), normalize)?,
                TransformKind::Combine { query: q, .. }
                | TransformKind::Join { query: q, .. }
                | TransformKind::Diff { query: q } => {
                    visit(q, typing, at.field("query"), normalize)?
                }
                TransformKind::Extension { arguments, .. } => {
                    extension(arguments, typing, at.field("arguments"), normalize)?
                }
                _ => {}
            }
        }
        Ok(())
    })
}

fn condition(
    c: &mut Condition,
    typing: &Typing,
    path: NodePath,
    normalize: bool,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        let at = path.field("kind");
        let position = c.span.start;
        match &mut c.kind {
            ConditionKind::And { conditions } | ConditionKind::Or { conditions } => {
                for (index, c) in conditions.iter_mut().enumerate() {
                    condition(c, typing, at.field("conditions").index(index), normalize)?;
                }
            }
            ConditionKind::Not { condition: c } => {
                condition(c, typing, at.field("condition"), normalize)?
            }
            ConditionKind::Match { value, .. } => {
                values(typing, &at.field("field"), value, position, normalize)?;
            }
            ConditionKind::Compare {
                left: Expression::Field { .. },
                right: Expression::Value { value },
                ..
            } => values(typing, &at.field("left"), value, position, normalize)?,
            ConditionKind::Compare {
                left: Expression::Value { value },
                right: Expression::Field { .. },
                ..
            } => values(typing, &at.field("right"), value, position, normalize)?,
            ConditionKind::Between {
                left: Expression::Field { .. },
                from,
                to,
                ..
            } => {
                values(typing, &at.field("left"), from, position, normalize)?;
                values(typing, &at.field("left"), to, position, normalize)?;
            }
            ConditionKind::In { query: q, .. } | ConditionKind::Order { query: q, .. } => {
                visit(q, typing, at.field("query"), normalize)?
            }
            ConditionKind::Exists {
                condition: Some(c), ..
            }
            | ConditionKind::Count {
                condition: Some(c), ..
            } => condition(c, typing, at.field("condition"), normalize)?,
            ConditionKind::Sequence { steps, .. } => {
                for (index, c) in steps.iter_mut().enumerate() {
                    condition(c, typing, at.field("steps").index(index), normalize)?;
                }
            }
            ConditionKind::Extension { arguments, .. } => {
                extension(arguments, typing, at.field("arguments"), normalize)?
            }
            _ => {}
        }
        Ok(())
    })
}

fn extension(
    arguments: &mut [ExtensionArgument],
    typing: &Typing,
    path: NodePath,
    normalize: bool,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        for (index, argument) in arguments.iter_mut().enumerate() {
            let at = path.index(index);
            match argument {
                ExtensionArgument::Condition(c) => {
                    condition(c, typing, at.field("condition"), normalize)?
                }
                ExtensionArgument::Query(q) => visit(q, typing, at.field("query"), normalize)?,
                ExtensionArgument::Optional(arguments) => {
                    extension(arguments, typing, at.field("optional"), normalize)?
                }
                ExtensionArgument::List(arguments) => {
                    extension(arguments, typing, at.field("list"), normalize)?
                }
                _ => {}
            }
        }
        Ok(())
    })
}

/// Reference values normalize before binding an identity; numeric values are
/// checked against the declared field. Other kinds retain their own rules.
fn values(
    typing: &Typing,
    operand: &NodePath,
    value: &mut Value,
    position: usize,
    normalize: bool,
) -> Result<(), QueryError> {
    query_language::depth::deeper(|| {
        let Some(typed) = typing.operand(operand) else {
            return Ok(());
        };
        if normalize {
            if let Origin::Catalog { unit, field } = &typed.result.origin
                && let Some(unit) = Unit::from_bound_name(unit)
                && let Some(field) = crate::fields::lookup(unit, field)
                && field.declaration().reference_argument
            {
                return reference_values(value, position, field);
            }
            return Ok(());
        }
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
                    values(typing, operand, member, position, normalize)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    })
}

/// Reference fields keep their storage identity while the query uses human names.
fn reference_values(
    value: &mut Value,
    position: usize,
    field: crate::fields::FieldId,
) -> Result<(), QueryError> {
    match value {
        Value::Scalar { text, .. } => match polylogue_model::Reference::parse(text) {
            Ok(polylogue_model::Reference::Assertion(revision))
                if field == crate::fields::FieldId::AnnotationsRoot =>
            {
                *text = revision.root;
            }
            // Root IDs remain opaque. Only an annotation reference names the
            // family through its public reference instead of its raw root ID.
            _ if field == crate::fields::FieldId::AnnotationsRoot => {}
            Ok(reference) => *text = reference.stored_form(),
            Err(error @ polylogue_model::ReferenceError::StoredSpelling { .. }) => {
                return Err(QueryError::Invalid {
                    message: error.to_string(),
                    position,
                });
            }
            _ => {}
        },
        Value::Set { values } => {
            for value in values {
                reference_values(value, position, field)?;
            }
        }
        _ => {}
    }
    Ok(())
}
