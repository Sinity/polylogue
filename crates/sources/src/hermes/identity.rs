/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use polylogue_model::Json;
use polylogue_model::JsonExt;
use std::path::{Component, Path, PathBuf};

/// The Hermes profile a `state.db` belongs to, by name: `default` for the
/// home itself (`$HERMES_HOME`, default `~/.hermes`) and `<name>` for
/// `<home>/profiles/<name>/`. Neither the machine nor where the home lives
/// is part of it, so moving the home or renaming the machine keeps every
/// session ID.
pub(super) fn profile(source: Option<&Json>) -> Result<Option<String>, &'static str> {
    let Some(source) = source.filter(|source| !source.is_null()) else {
        return Ok(None);
    };
    let raw = source.str("path").ok_or("missing source path")?;
    let mut path = PathBuf::new();
    for component in Path::new(raw).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                path.pop();
            }
            component => path.push(component.as_os_str()),
        }
    }
    if !path.is_absolute() || path.file_name().is_none_or(|name| name != "state.db") {
        return Err("not an absolute Hermes state.db source");
    }
    let home = path.parent().ok_or("missing Hermes profile root")?;
    let in_profiles = home
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|name| name == "profiles");
    let name = match home.file_name().and_then(|name| name.to_str()) {
        Some(name) if in_profiles => component(name),
        _ => "default".to_owned(),
    };
    Ok(Some(name))
}

// Escape the category and occurrence delimiters without changing the source bytes.
pub(super) fn component(value: &str) -> String {
    value
        .replace('%', "%25")
        .replace(':', "%3A")
        .replace('~', "%7E")
}
