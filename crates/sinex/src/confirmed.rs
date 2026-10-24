//! One existing confirmed-descriptor fold for restore and held machine checks.
use super::*;

impl Sinex {
    pub(super) async fn confirmed(
        &self,
        archive: &str,
        mut catalog: sinex_rows::Catalog,
    ) -> Result<sinex_rows::Selection> {
        let js = jetstream::new(self.client.clone());
        let stream = js
            .get_stream(self.subjects.confirmed_stream())
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let consumer = stream
            .create_consumer(jetstream::consumer::pull::OrderedConfig {
                filter_subject: self.subjects.confirmed_event("polylogue.rows"),
                deliver_policy: jetstream::consumer::DeliverPolicy::All,
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        // Read up to the last unit on the stream now; later ones arrive with
        // the next pass.
        let last = match stream
            .get_last_raw_message_by_subject(&self.subjects.confirmed_event("polylogue.rows"))
            .await
        {
            Ok(m) => m.sequence,
            Err(e) if e.kind() == jetstream::stream::LastRawMessageErrorKind::NoMessageFound => 0,
            Err(e) => return Err(Error::Unreachable(e.to_string())),
        };
        let mut messages = consumer
            .messages()
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let mut done = last == 0;
        while !done {
            let message = tokio::time::timeout(ACK_WAIT, messages.next())
                .await
                .map_err(|_| Error::Unreachable("the confirmed stream stopped".into()))?
                .ok_or_else(|| Error::Unreachable("the confirmed stream closed".into()))?
                .map_err(|e| Error::Unreachable(e.to_string()))?;
            let at = message
                .info()
                .map_err(|e| Error::Unreachable(e.to_string()))?
                .stream_sequence;
            done = at >= last;
            let confirmed: sinex_protocol::Confirmed = serde_json::from_slice(&message.payload)?;
            let sinex_protocol::Confirmed::Event(event) = confirmed else {
                continue;
            };
            let mut payload = event.payload.ok_or_else(|| {
                Error::Invalid(format!("{} arrived without its payload", event.id))
            })?;
            let excision = payload
                .as_object_mut()
                .and_then(|object| object.remove("excision"));
            let unit: PolylogueRows = serde_json::from_value(payload)?;
            if unit.archive == archive {
                let excision = excision.map(serde_json::from_value).transpose()?;
                catalog.push_evidence(
                    unit.position,
                    &RestoreUnit {
                        unit: Unit {
                            commit_seq: unit.local_seq,
                            kind: unit.unit,
                            key: unit.key,
                        },
                        batch: Some(unit.batch),
                        bytes: None,
                        rows: unit.rows,
                        blobs: unit.blobs,
                        excision,
                    },
                    &event.id.to_string(),
                )?;
            }
        }
        Ok(catalog.finish()?)
    }
}
