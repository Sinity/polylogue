//! Export callback capture is selected by durable queue keys; sidecars and
//! material frames retain the original bytes.
use super::*;
use polylogue_store::primary::{
    ExportReadingKey,
    queue::{Lookup, Order},
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ExportBinding {
    pub key: ExportReadingKey,
    /// None identifies the material's product stream, otherwise a sidecar slot.
    pub captured: Option<usize>,
}

pub(super) fn bindings(frame: &Ingress, capture: Option<&Capture>) -> Result<Vec<ExportBinding>> {
    let mut result = capture
        .map(capture_bindings)
        .transpose()?
        .unwrap_or_default();
    if let Ingress::Material { material, .. } = frame
        && let Some(value) = material.manifest.details.get("polylogue_export_key")
    {
        if !result.is_empty() || capture.is_some() {
            return invalid("export material has conflicting capture layouts");
        }
        let key: ExportReadingKey = serde_json::from_value(value.clone())?;
        key.validate()?;
        result.push(ExportBinding {
            key,
            captured: None,
        });
    }
    Ok(result)
}

pub(super) fn capture_bindings(capture: &Capture) -> Result<Vec<ExportBinding>> {
    let mut result = Vec::new();
    if let Capture::Input(input) = capture {
        for (index, reading) in input.parsed.iter().enumerate() {
            if reading.delivery.is_some() {
                let Some(key) = &reading.export_key else {
                    continue;
                };
                if key.validate().is_err()
                    || key.input.id != input.rows.input
                    || key.input.format != input.rows.format
                    || key.reading_version != reading.reading_version
                {
                    continue;
                }
                result.push(ExportBinding {
                    key: key.clone(),
                    captured: Some(index),
                });
            }
        }
    }
    Ok(result)
}

pub(super) fn unknown(capture: &Capture, bindings: &[ExportBinding]) -> bool {
    let Capture::Input(input) = capture else {
        return false;
    };
    input.parsed.iter().enumerate().any(|(index, reading)| {
        (reading.delivery.is_some() || reading.export_key.is_some())
            && !bindings
                .iter()
                .any(|binding| binding.captured == Some(index))
    })
}

impl Spool {
    pub fn next_export(
        &self,
        key: &ExportReadingKey,
        after: Option<u64>,
        through: u64,
    ) -> Result<Option<Prepared>> {
        let identity = key.identity().to_string();
        self.queue
            .lookup(
                Lookup::Export {
                    input: &key.input.id,
                    format: &key.input.format,
                    key: Some(&identity),
                },
                after,
                Some(through),
                Order::OldestFirst,
                1,
            )?
            .into_iter()
            .next()
            .map(|entry| self.prepared(entry))
            .transpose()
    }

    /// Refuses export progress for `key` while capture of the same input
    /// holds an export reading that is bound to no valid key: its callback
    /// may belong to this export, and its body cannot be checked against it.
    pub fn require_known_exports(&self, key: &ExportReadingKey, through: u64) -> Result<()> {
        let mut after = None;
        loop {
            let entries = self.queue.lookup(
                Lookup::Export {
                    input: &key.input.id,
                    format: &key.input.format,
                    key: None,
                },
                after,
                Some(through),
                Order::OldestFirst,
                64,
            )?;
            if entries.is_empty() {
                return Ok(());
            }
            for entry in entries {
                after = Some(entry.order);
                let prepared = self.prepared(entry)?;
                if !matches!(prepared.capture(), Some(Capture::Input(_))) {
                    continue;
                }
                let capture = self.capture(&prepared.identity)?;
                let bindings = capture_bindings(&capture)?;
                if unknown(&capture, &bindings) {
                    return Err(Error::Refused {
                        name: key.identity().to_string(),
                        reason: "pending capture of this input holds an export reading without a valid export key".into(),
                    });
                }
            }
        }
    }

    pub fn export_material_key(&self, identity: &str) -> Result<Option<ExportReadingKey>> {
        let entry = self.required(identity)?;
        let frame = self.decode_frame(identity)?;
        let actual = bindings(&frame, None)?;
        let expected = entry
            .exports()
            .iter()
            .filter(|binding| binding.captured.is_none())
            .cloned()
            .collect::<Vec<_>>();
        if actual != expected {
            return invalid("export key differs from its immutable material frame");
        }
        Ok(actual.into_iter().next().map(|binding| binding.key))
    }
}
