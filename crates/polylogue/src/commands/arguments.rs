//! Command-line argument declarations read from a request type's schema.
//!
//! A request field is declared once, on its type: its doc comment is its
//! help, its serde default its default, its type its kind and choices. How
//! it is written on the command line is the schema key `argument_form`
//! (`positional`, `words` or `rest`, set with `#[schemars(extend(...))]`);
//! without one a boolean is a flag, a list a repeated `--name` whose values
//! may also be comma-separated, and anything else `--name VALUE`. The key
//! `unlisted` leaves a field out of the MCP tool list.
use command_surface::{ArgumentForm, ArgumentSpec};
use serde_json::Value;

/// One argument per property of `schema`, in field order. `calling_process`
/// marks options that act on the calling machine and stay out of the request.
pub(crate) fn from_schema(schema: &schemars::Schema, calling_process: bool) -> Vec<ArgumentSpec> {
    let root = schema.as_value();
    let Some(properties) = root.get("properties").and_then(Value::as_object) else {
        return Vec::new();
    };
    properties
        .iter()
        .map(|(name, property)| {
            let form = match property.get("argument_form") {
                Some(form) => serde_json::from_value(form.clone())
                    .unwrap_or_else(|_| panic!("{name}: argument_form names an argument form")),
                None => match value_type(property, root) {
                    Some("boolean") => ArgumentForm::Flag,
                    Some("array") => ArgumentForm::Repeated,
                    _ => ArgumentForm::Named,
                },
            };
            let mut argument = ArgumentSpec::new(
                name.as_str(),
                form,
                property
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            argument.calling_process = calling_process;
            argument.listed = property.get("unlisted") != Some(&Value::Bool(true));
            argument.value_delimiter = (form == ArgumentForm::Repeated).then_some(',');
            argument
        })
        .collect()
}

/// The one JSON type a property holds apart from `null`, following `$ref`
/// into `$defs` and a single non-null `anyOf` branch.
fn value_type<'a>(schema: &'a Value, root: &'a Value) -> Option<&'a str> {
    let schema = match schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| reference.strip_prefix("#/$defs/"))
    {
        Some(name) => root.get("$defs")?.get(name)?,
        None => schema,
    };
    if let Some(branches) = schema.get("anyOf").and_then(Value::as_array) {
        let mut types = branches
            .iter()
            .filter_map(|branch| value_type(branch, root))
            .filter(|kind| *kind != "null");
        return match (types.next(), types.next()) {
            (Some(kind), None) => Some(kind),
            _ => None,
        };
    }
    match schema.get("type")? {
        Value::String(kind) => Some(kind),
        Value::Array(kinds) => {
            let mut kinds = kinds
                .iter()
                .filter_map(Value::as_str)
                .filter(|kind| *kind != "null");
            match (kinds.next(), kinds.next()) {
                (Some(kind), None) => Some(kind),
                _ => None,
            }
        }
        _ => None,
    }
}
