//! Direct documents stream canonically ordered installed leaves; model pages admit the complete
//! opaque object before retaining it for canonical key ordering and framing.
use super::{Encoder, METADATA, RawJson, invalid};
use polylogue_model::NativeId;
use std::{
    collections::BTreeMap,
    io::{self, Write},
};

pub(super) struct OpaqueFields {
    name: &'static str,
    started: bool,
    last: Option<String>,
    page: Option<(BTreeMap<NativeId, RawJson>, usize)>,
}
impl OpaqueFields {
    pub fn new(name: &'static str, model_page: bool) -> Self {
        Self {
            name,
            started: false,
            last: None,
            page: model_page.then(|| (BTreeMap::new(), 2)),
        }
    }
    pub fn push<W: Write>(&mut self, e: &mut Encoder<W>, key: &str, raw: &[u8]) -> io::Result<()> {
        if let Some((fields, bytes)) = &mut self.page {
            // Charge the bounded raw leaf before parsing or retaining it. The
            // final Metadata sink separately bounds privacy/escaping expansion.
            let key_bytes = serde_json::to_vec(key)?.len();
            let next = bytes
                .checked_add(key_bytes)
                .and_then(|n| n.checked_add(2))
                .and_then(|n| n.checked_add(raw.len()))
                .filter(|n| *n <= METADATA)
                .ok_or_else(|| invalid("selected opaque metadata exceeds admission"))?;
            let value = RawJson::parse(raw).map_err(io::Error::other)?;
            fields.insert(key.into(), value);
            *bytes = next;
        } else {
            if self.last.as_deref().is_some_and(|last| last >= key) {
                return Err(invalid("opaque field keys are not in canonical order"));
            }
            self.last = Some(key.to_owned());
            let value = RawJson::parse(raw).map_err(io::Error::other)?;
            if !self.started {
                e.syntax(b",")?;
                e.key(self.name)?;
                e.syntax(b"{")?;
                self.started = true;
            } else {
                e.syntax(b",")?;
            }
            e.key(key)?;
            e.retained(&value, false)?;
        }
        Ok(())
    }
    pub fn finish<W: Write>(self, e: &mut Encoder<W>) -> io::Result<()> {
        if let Some((fields, _)) = self.page {
            if !fields.is_empty() {
                e.syntax(b",")?;
                e.key(self.name)?;
                e.opaque(&RawJson::Object(fields))?;
            }
        } else if self.started {
            e.syntax(b"}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "opaque_fields_tests.rs"]
mod tests;
