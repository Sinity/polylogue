use super::*;
use crate::primary::{
    Capture, CapturedInput, Confirmed, NamedProductSelector, PendingCapture, PreparedBody,
    PreparedInput, PreparedWrite, Product, ProductWrite, ReadingProducts, Sinex,
};
use polylogue_model::{BlobHash, authored::PendingAuthored, user::UserOutcome};
use std::{path::PathBuf, sync::Arc};

struct PendingProducer(PathBuf);
impl PendingProducer {
    fn writes(&self) -> Result<Option<(String, Vec<ProductWrite>)>> {
        if !self.0.exists() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_slice(&std::fs::read(&self.0)?)?))
    }
    fn unavailable<T>() -> Result<T> {
        Err(StoreError::Sinex("synthetic offline producer".into()))
    }
}
impl Sinex for PendingProducer {
    fn prepare_input(&self, _: &CapturedInput, _: PreparedBody) -> Result<PreparedInput> {
        Self::unavailable()
    }
    fn prepare_blob(&self, _: BlobHash, _: u64, _: PreparedBody) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn pending_captures(&self) -> Result<Vec<PendingCapture>> {
        Ok(vec![])
    }
    fn capture(&self, _: &str) -> Result<Capture> {
        Self::unavailable()
    }
    fn complete_captures(&self, _: &[String], _: &[ProductWrite]) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn prepare_products(&self, _: &[ProductWrite]) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn prepare_request(
        &self,
        _: &polylogue_model::authored::PreparedRequest,
    ) -> Result<UserOutcome> {
        Self::unavailable()
    }
    fn visit_pending_requests(
        &self,
        _: &mut dyn FnMut(&PendingAuthored) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }
    fn visit_pending_blobs(&self, _: &mut dyn FnMut(&crate::NewBlob) -> Result<()>) -> Result<()> {
        Ok(())
    }
    fn prepare_reading(
        &self,
        _: &str,
        _: &str,
        _: &mut ReadingProducts<'_>,
    ) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn visit_named_products(
        &self,
        selector: &NamedProductSelector,
        visit: &mut dyn FnMut(&Product) -> Result<()>,
    ) -> Result<()> {
        if let Some((name, writes)) = self.writes()? {
            let selected = match selector {
                NamedProductSelector::Exact(expected) => *expected == name,
                NamedProductSelector::Prefix(prefix) => name.starts_with(prefix),
            };
            if selected {
                for write in writes {
                    visit(&write.product)?;
                }
            }
        }
        Ok(())
    }
    fn prepare_named_products(
        &self,
        identity: &str,
        writes: &[ProductWrite],
    ) -> Result<PreparedWrite> {
        if let Some((prior, existing)) = self.writes()? {
            assert_eq!(prior, identity);
            assert_eq!(existing, writes);
        } else {
            let file = std::fs::File::create(&self.0)?;
            serde_json::to_writer(&file, &(identity, writes))?;
            file.sync_all()?;
            std::fs::File::open(self.0.parent().unwrap())?.sync_all()?;
        }
        Ok(PreparedWrite {
            identity: identity.into(),
            pending: true,
        })
    }
    fn discard_confirmed(&self, _: &[String], _: i64, _: i64) -> Result<()> {
        Ok(())
    }
    fn confirmed(&self, _: i64, _: &str, _: Option<&str>) -> Result<Vec<Confirmed>> {
        Self::unavailable()
    }
    fn acknowledge(&self, _: i64) -> Result<()> {
        Self::unavailable()
    }
}

#[test]
fn completed_run_keeps_its_original_scope_in_durable_custody_during_cache_reset() {
    let dir = tempfile::tempdir().unwrap();
    let producer = Arc::new(PendingProducer(dir.path().join("named.json")));
    let object = question("sessions");
    let mut expected = run(
        &object,
        "scope-run",
        7,
        ChangePosition {
            generation: "original-generation".into(),
            index: 5,
            archive: 4,
        },
    );
    expected.replay.scope_fingerprint = Some("original-scope".into());
    for _ in 0..2 {
        let (archive, _lock) =
            crate::Archive::open_primary(&dir.path().join("cache"), &dir.path().join("bytes"))
                .unwrap();
        let conn = crate::archive::open_write(&archive.index_db(), 1024).unwrap();
        conn.execute("INSERT INTO meta(key,value) VALUES ('confirmed_scope_pending','replacement-scope') ON CONFLICT(key) DO UPDATE SET value=excluded.value", []).unwrap();
        let writer = crate::Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        assert!(matches!(
            writer.query_object(object.clone()),
            Err(StoreError::ScopeChanged)
        ));
        assert_eq!(
            writer
                .record_query_run(expected.clone())
                .unwrap()
                .recv()
                .unwrap()
                .unwrap(),
            QueryCustody::Queued
        );
        let (_, writes) = producer.writes().unwrap().unwrap();
        let [write] = writes.as_slice() else {
            panic!("one frozen query run expected");
        };
        assert!(write.parents.is_empty());
        let Product::QueryRun { run } = &write.product else {
            panic!("query run expected");
        };
        assert_eq!(**run, expected);
    }
}
