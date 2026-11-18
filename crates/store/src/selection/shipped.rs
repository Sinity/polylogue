//! Prepared built-in definitions, shared by readers with the same origin catalog.
use super::*;
use polylogue_query::{Bindings, SavedQuery};
use std::sync::Mutex;

type Definitions = BTreeMap<String, Vec<SavedQuery>>;
static SHIPPED: Mutex<Vec<(Vec<String>, Definitions)>> = Mutex::new(Vec::new());

pub fn shipped_bindings(origins: &[String]) -> Result<Definitions, SelectionError> {
    let mut cache = SHIPPED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, definitions)) = cache.iter().find(|(key, _)| key == origins) {
        return Ok(definitions.clone());
    }
    let bindings = Bindings {
        origins: origins.to_vec(),
        ..Default::default()
    };
    let mut definitions = Definitions::new();
    for shipped in polylogue_model::saved_queries::SHIPPED_QUERIES {
        let plan = polylogue_query::plan::prepare(shipped.text, &bindings, |_| Ok(()))?;
        let object = plan.query_object()?;
        let nodes = polylogue_query::plan::saved_query_bindings(&object)?;
        definitions.insert(shipped.name.into(), nodes[&object.id.to_string()].clone());
        definitions.extend(nodes);
    }
    cache.push((origins.to_vec(), definitions.clone()));
    Ok(definitions)
}
