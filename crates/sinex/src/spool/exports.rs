//! Rebuildable references to exact export callback custody. The immutable
//! sidecar or material frame owns the key; the index never supplies it.
use super::*;
use polylogue_store::primary::ExportReadingKey;

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
            return invalid("export material has conflicting custody layouts");
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

pub(super) fn insert(spool: &Spool, prepared: &Prepared) -> Result<()> {
    // Every captured input is indexed by its source name, including those
    // whose sidecar cannot supply a valid export key.
    if let Some(material) = prepared.material()
        && let Some(Capture::Input(input)) = &material.capture
    {
        let path = spool.index(
            "export-inputs",
            &input_key(&input.rows.input, &input.rows.format)?,
        );
        insert_order(&path, prepared)?;
    }
    for binding in prepared.exports() {
        let path = spool.index("export-readings", &binding.key.identity().to_string());
        insert_order(&path, prepared)?;
    }
    Ok(())
}

fn input_key(input: &str, format: &str) -> Result<String> {
    Ok(hash(&serde_json::to_vec(&(input, format))?))
}

fn insert_order(root: &Path, prepared: &Prepared) -> Result<()> {
    let mut path = root.to_path_buf();
    for byte in &prepared.order.to_be_bytes()[..7] {
        path.push(format!("{byte:02x}"));
    }
    fs::create_dir_all(&path)?;
    path.push(format!("{:02x}", prepared.order.to_be_bytes()[7]));
    write_index(&path, &prepared.identity)
}

impl Spool {
    pub fn next_export(
        &self,
        key: &ExportReadingKey,
        after: Option<u64>,
        through: u64,
    ) -> Result<Option<Prepared>> {
        self.ensure_indexes()?;
        let path = self.index("export-readings", &key.identity().to_string());
        if !path.try_exists()? {
            return Ok(None);
        }
        let mut found = None;
        raw_ranges::walk(&path, 0, 0, after, through, 1, &mut |order, identity| {
            let entry = self.required(identity)?;
            if entry.order != order || !entry.exports().iter().any(|binding| &binding.key == key) {
                return invalid("export index differs from its immutable custody key");
            }
            found = Some(entry);
            Ok(())
        })?;
        Ok(found)
    }

    /// Refuses export progress for `key` while custody of the same input
    /// holds an export reading that is bound to no valid key: its callback
    /// may belong to this export, and its body cannot be checked against it.
    pub fn require_known_exports(&self, key: &ExportReadingKey, through: u64) -> Result<()> {
        self.ensure_indexes()?;
        let path = self.index(
            "export-inputs",
            &input_key(&key.input.id, &key.input.format)?,
        );
        if !path.try_exists()? {
            return Ok(());
        }
        raw_ranges::walk(&path, 0, 0, None, through, usize::MAX, &mut |order, identity| {
            let entry = self.required(identity)?;
            let input_matches = entry.capture().is_some_and(|capture| matches!(capture,
                Capture::Input(input) if input.rows.input == key.input.id && input.rows.format == key.input.format));
            if entry.order != order || !input_matches {
                return invalid("export input index differs from its immutable custody");
            }
            if entry.phase.is_refused() {
                return Ok(());
            }
            let capture = self.capture(identity)?;
            let bindings = capture_bindings(&capture)?;
            if !unknown(&capture, &bindings) {
                return Ok(());
            }
            Err(Error::Refused {
                name: key.identity().to_string(),
                reason: "pending custody of this input holds an export reading without a valid export key".into(),
            })
        })
        .map(|_| ())
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
