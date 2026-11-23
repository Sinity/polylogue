//! Validates signal annotations into rule definitions for the evaluator; a
//! definition is validated content, not an active revision.
use polylogue_model::annotation::{Annotation, AnnotationKind, values};

/// Validated content is distinct from an active evaluator revision. Neither a
/// stated historical author nor a local index counter grants activation.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Definition {
    pub family: String,
    pub rule: shared_signals::Rule,
}

pub(super) fn definition(
    annotation: &Annotation,
    configuration: &shared_signals::Configuration,
) -> Result<Definition, String> {
    if annotation.fact.kind != AnnotationKind::Signal {
        return Err("annotation is not a signal".into());
    }
    let values::Value::Signal(value) = annotation.fact.validate().map_err(|e| e.to_string())?
    else {
        return Err("signal annotation has another value shape".into());
    };
    value
        .fields
        .validate(configuration)
        .map_err(|e| e.to_string())?;
    Ok(Definition {
        family: annotation.fact.root.clone(),
        rule: value.fields,
    })
}
