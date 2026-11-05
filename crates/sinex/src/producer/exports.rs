//! Bound export custody uses the ordinary spool. No callback hash or local
//! queue position supplies source acceptance or native authority.
use super::*;
use polylogue_store::primary::{CapturedReading, ExportReadingKey};
use std::io::BufRead;

impl Sinex {
    pub(crate) fn visit_pending_export_readings(
        &self,
        key: &ExportReadingKey,
        visit: &mut dyn FnMut(&CapturedReading) -> polylogue_store::Result<()>,
    ) -> Result<()> {
        key.validate()?;
        let through = self.queue()?.raw_cut()?;
        self.queue()?.require_known_exports(key, through)?;
        let mut after = None;
        loop {
            let Some(entry) = self.queue()?.next_export(key, after, through)? else {
                return Ok(());
            };
            after = Some(entry.order);
            if entry.unresolved || entry.phase.is_refused() {
                continue;
            }
            if entry.capture().is_some() {
                let pinned = self.queue()?.pin_capture(&entry.identity)?;
                let Capture::Input(input) = pinned.decode()? else {
                    return Err(Error::Invalid(
                        "export index references a non-input capture".into(),
                    ));
                };
                for reading in &input.parsed {
                    if reading.export_key.as_ref() == Some(key) {
                        reading.clone().into_step().validate_export()?;
                        visit(reading)?;
                    }
                }
            } else {
                let (reader, ndjson) = {
                    let queue = self.queue()?;
                    if queue.export_material_key(&entry.identity)?.as_ref() != Some(key) {
                        return Err(Error::Invalid(
                            "export custody has another material key".into(),
                        ));
                    }
                    let Ingress::Material { material, .. } = queue.decode_frame(&entry.identity)?
                    else {
                        return Err(Error::Invalid(
                            "export custody has no product material".into(),
                        ));
                    };
                    match material.parser.as_str() {
                        "polylogue-reading" => (queue.body(&entry.identity)?, true),
                        "polylogue-products" => (queue.body(&entry.identity)?, false),
                        _ => {
                            return Err(Error::Invalid("unsupported export custody layout".into()));
                        }
                    }
                };
                if ndjson {
                    let mut group = Vec::new();
                    for line in std::io::BufReader::new(reader).lines() {
                        let write: ProductWrite = crate::spool::decode_json(line?.as_bytes())?;
                        crate::products::require_reading(&write.product, key)?;
                        let complete = matches!(&write.product, Product::ParseProgress { .. });
                        group.push(write.product);
                        if complete {
                            if group.iter().any(|product| {
                                product
                                    .reading()
                                    .is_some_and(|reading| reading.delivery.is_some())
                            }) {
                                visit(&CapturedReading::from_product_group(key, &group)?)?;
                            }
                            group.clear();
                        }
                    }
                    if !group.is_empty() {
                        return Err(Error::Invalid(
                            "export custody ends before its callback control".into(),
                        ));
                    }
                } else {
                    let writes: Vec<ProductWrite> = crate::spool::decode_json(reader)?;
                    for write in &writes {
                        crate::products::require_export(&write.product, key)?;
                    }
                    visit(&CapturedReading::from_product_group(
                        key,
                        writes.iter().map(|write| &write.product),
                    )?)?;
                }
            }
        }
    }

    pub(crate) fn prepare_export_products(
        &self,
        key: &ExportReadingKey,
        writes: &[ProductWrite],
    ) -> Result<PreparedWrite> {
        for write in writes {
            crate::products::require_export(&write.product, key)?;
        }
        // This also verifies the semantic body, original event IDs, and the
        // single callback control before any durable semantic publication.
        CapturedReading::from_product_group(key, writes.iter().map(|write| &write.product))?;
        let content = blake3::hash(&serde_json::to_vec(writes)?);
        let identity = format!("export:{}:{}", key.identity(), content.to_hex());
        self.prepare_products_bound(&identity, writes, Some(key))
    }
}
