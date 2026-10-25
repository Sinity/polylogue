use polylogue_model::{BlobHash, Json};
use std::path::{Component, Path, PathBuf};

pub(super) fn profile(source: Option<&Json>) -> Result<Option<String>, &'static str> {
    let Some(source) = source.filter(|source| !source.is_null()) else {
        return Ok(None);
    };
    let machine = source
        .get("machine")
        .and_then(Json::as_str)
        .filter(|machine| !machine.is_empty())
        .ok_or("missing native machine")?;
    let raw = source
        .get("path")
        .and_then(Json::as_str)
        .ok_or("missing native path")?;
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
    let root = path.parent().ok_or("missing native profile root")?;
    let hash = BlobHash::of(format!("{machine}\0{}", root.display()).as_bytes()).to_string();
    Ok(Some(hash[..16].to_owned()))
}

pub(super) fn hash(value: &Json) -> String {
    let mut value = value.clone();
    value.sort_all_objects();
    BlobHash::of(value.to_string().as_bytes()).to_string()
}

// Escape the category and occurrence delimiters without changing native bytes.
pub(super) fn component(value: &str) -> String {
    value
        .replace('%', "%25")
        .replace(':', "%3A")
        .replace('~', "%7E")
}
