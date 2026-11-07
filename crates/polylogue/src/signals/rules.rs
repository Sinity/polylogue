use polylogue_model::assertion::{Assertion, AssertionKind, values};

/// Validated content is distinct from an active evaluator revision. Neither a
/// historical author claim nor the local projection counter grants activation.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Definition {
    pub family: String,
    pub rule: shared_signals::Rule,
}

pub(super) fn definition(
    assertion: &Assertion,
    configuration: &shared_signals::Configuration,
) -> Result<Definition, String> {
    if assertion.fact.kind != AssertionKind::Signal {
        return Err("assertion is not a signal".into());
    }
    let values::Value::Signal(value) = assertion.fact.validate().map_err(|e| e.to_string())? else {
        return Err("signal assertion has another value shape".into());
    };
    value
        .fields
        .validate(configuration)
        .map_err(|e| e.to_string())?;
    Ok(Definition {
        family: assertion.fact.root.clone(),
        rule: value.fields,
    })
}
