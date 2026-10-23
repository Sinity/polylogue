use super::*;
use crate::Observation;
use serde::{Deserialize, Serialize};
use std::path::Component;

/// Identity from the native database's first observation, never its export path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Binding {
    machine: String,
    path: String,
}

impl Binding {
    pub(super) fn observed(observation: &Observation) -> Result<Self, ExportError> {
        let mut path = PathBuf::new();
        for component in Path::new(&observation.path).components() {
            match component {
                Component::CurDir => {}
                Component::ParentDir => {
                    path.pop();
                }
                component => path.push(component.as_os_str()),
            }
        }
        if observation.machine.is_empty() || !path.is_absolute() {
            return Err(ExportError::Invalid(
                "logical source needs a machine and absolute native path".into(),
            ));
        }
        Ok(Self {
            machine: observation.machine.clone(),
            path: path.display().to_string(),
        })
    }
}
