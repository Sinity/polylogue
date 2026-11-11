use super::*;
use crate::primary::{
    Capture, CapturedInput, Confirmed, NamedProductSelector, PendingCapture, PreparedBody,
    PreparedInput, PreparedWrite, Product, ProductWrite, ReadingProducts, Sinex,
};
use polylogue_model::{BlobHash, authored::PendingAuthored, user::UserOutcome};
use std::{path::PathBuf, sync::Arc};

struct PendingProducer(PathBuf);

#[test]
fn completed_history_keeps_question_lookup_and_one_durable_run_group() {
    let dir = tempfile::tempdir().unwrap();
    let producer = Arc::new(PendingProducer(dir.path().join("named.json")));
    let object = question("sessions");
    let expected = run(
        &object,
        "completed-pending-history",
        7,
        ChangePosition {
            generation: "synthetic-generation".into(),
            index: 5,
            archive: 4,
        },
    );
    let completed = outcome(&expected, 5);
    for _ in 0..2 {
        let (archive, _lock) =
            crate::Archive::open_primary(&dir.path().join("cache"), &dir.path().join("bytes"))
                .unwrap();
        let writer = crate::Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        writer
            .sender()
            .unwrap()
            .reserve_history()
            .unwrap()
            .record(object.clone(), expected.clone(), completed.clone())
            .unwrap();
        assert!(writer.flush_rows().unwrap().errors.is_empty());
        let pool = crate::ReadPool::open(&archive).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader
                .query_object(&object.id.to_string(), Some(producer.as_ref()))
                .unwrap()
                .unwrap()
                .value,
            object
        );
        assert_eq!(
            reader
                .query_run(&expected.id, Some(producer.as_ref()))
                .unwrap()
                .unwrap()
                .value,
            expected
        );
        assert_eq!(
            reader
                .query_run_outcome(&expected.id, Some(producer.as_ref()))
                .unwrap()
                .unwrap()
                .value,
            completed
        );
        let retained = producer.writes().unwrap();
        assert_eq!(retained.len(), 2);
        let group = &retained[&format!("query-run:{}", expected.id)];
        assert_eq!(group.len(), 2);
        assert!(matches!(&group[0].product, Product::QueryRun { run } if **run == expected));
        assert!(
            matches!(&group[1].product, Product::QueryRunOutcome { outcome } if **outcome == completed)
        );
    }
}

impl PendingProducer {
    fn writes(&self) -> Result<BTreeMap<String, Vec<ProductWrite>>> {
        if !self.0.exists() {
            return Ok(BTreeMap::new());
        }
        Ok(serde_json::from_slice(&std::fs::read(&self.0)?)?)
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
    fn prepare_input_group(
        &self,
        _: &crate::primary::CapturePlan,
        _: &mut crate::primary::CaptureMembers<'_>,
    ) -> Result<crate::primary::CaptureReceipt> {
        Self::unavailable()
    }
    fn pending_source(
        &self,
        _: &crate::primary::SourceSelector,
        _: &str,
    ) -> Result<crate::primary::PendingSource> {
        Self::unavailable()
    }
    fn pending_group_headers(
        &self,
        _: &str,
        _: Option<u64>,
        _: usize,
    ) -> Result<crate::primary::PendingGroupPage> {
        Self::unavailable()
    }
    fn raw_acquisition_cursor(
        &self,
        _: &crate::primary::RawAcquisitionQuery,
    ) -> Result<Option<crate::primary::RawAcquisitionCursor>> {
        Ok(None)
    }
    fn capture_body(&self, _: &str) -> Result<Vec<PreparedBody>> {
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
    fn prepare_export_products(
        &self,
        _: &crate::primary::ExportReadingKey,
        _: &[crate::primary::ProductWrite],
    ) -> Result<crate::primary::PreparedWrite> {
        Self::unavailable()
    }
    fn visit_pending_export_readings(
        &self,
        _: &crate::primary::ExportReadingKey,
        _: &mut dyn FnMut(&crate::primary::CapturedReading) -> Result<()>,
    ) -> Result<()> {
        Ok(())
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
        _: &crate::primary::ExportReadingKey,
        _: &mut ReadingProducts<'_>,
    ) -> Result<PreparedWrite> {
        Self::unavailable()
    }
    fn visit_named_products(
        &self,
        selector: &NamedProductSelector,
        visit: &mut dyn FnMut(&Product) -> Result<()>,
    ) -> Result<()> {
        for (name, writes) in self.writes()? {
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
        let mut retained = self.writes()?;
        if let Some(existing) = retained.get(identity) {
            if existing != writes {
                return Err(StoreError::Conflict(identity.into()));
            }
        } else {
            retained.insert(identity.into(), writes.to_vec());
            let file = std::fs::File::create(&self.0)?;
            serde_json::to_writer(&file, &retained)?;
            file.sync_all()?;
            std::fs::File::open(self.0.parent().unwrap())?.sync_all()?;
        }
        Ok(PreparedWrite {
            identity: identity.into(),
            pending: true,
        })
    }
    fn confirmed(&self, _: &crate::primary::ReplayState, _: &str) -> Result<Vec<Confirmed>> {
        Self::unavailable()
    }
    fn acknowledge(
        &self,
        _: &crate::primary::ReplayState,
        _: &dyn Fn(&str) -> Result<bool>,
    ) -> Result<()> {
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
    expected
        .binding
        .original_mut()
        .unwrap()
        .replay
        .scope_fingerprint = Some("original-scope".into());
    expected
        .binding
        .original_mut()
        .unwrap()
        .replay
        .context
        .versions
        .insert("scope".into(), "original-scope".into());
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
            writer.begin_query_run(expected.clone()).unwrap(),
            QueryCustody::Queued
        );
        let retained = producer.writes().unwrap();
        let writes = retained.get(&format!("query-run:{}", expected.id)).unwrap();
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

#[test]
fn queued_begin_and_outcome_remain_distinct_and_lookup_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let producer = Arc::new(PendingProducer(dir.path().join("named.json")));
    let object = question("sessions");
    let mut expected = run(
        &object,
        "pending-begin",
        7,
        ChangePosition {
            generation: "synthetic-generation".into(),
            index: 5,
            archive: 4,
        },
    );
    expected.binding.original_mut().unwrap().accepted_request =
        Some(serde_json::json!({"id":"uncapped-caller".repeat(5000),"operation":"read"}));
    let completed = outcome(&expected, 5);
    for phase in 0..2 {
        let (archive, _lock) =
            crate::Archive::open_primary(&dir.path().join("cache"), &dir.path().join("bytes"))
                .unwrap();
        let writer = crate::Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        assert_eq!(
            writer.begin_query_run(expected.clone()).unwrap(),
            QueryCustody::Queued
        );
        let pool = crate::ReadPool::open(&archive).unwrap();
        assert_eq!(
            pool.reader()
                .unwrap()
                .query_run(&expected.id, Some(producer.as_ref()))
                .unwrap()
                .unwrap()
                .value,
            expected
        );
        if phase == 0 {
            assert!(
                pool.reader()
                    .unwrap()
                    .query_run_outcome(&expected.id, Some(producer.as_ref()))
                    .unwrap()
                    .is_none()
            );
        }
        drop(writer.finish_query_run(completed.clone()).unwrap());
        writer.flush_rows().unwrap();
        assert_eq!(
            pool.reader()
                .unwrap()
                .query_run_outcome(&expected.id, Some(producer.as_ref()))
                .unwrap()
                .unwrap()
                .value,
            completed
        );
        assert_eq!(
            pool.reader()
                .unwrap()
                .query_run(&expected.id, Some(producer.as_ref()))
                .unwrap()
                .unwrap()
                .value,
            expected
        );
        assert_eq!(producer.writes().unwrap().len(), 2);
    }
}

#[test]
fn confirmed_run_phases_reuse_canonical_custody_without_a_new_spool_write() {
    let dir = tempfile::tempdir().unwrap();
    let producer = Arc::new(PendingProducer(dir.path().join("named.json")));
    let (archive, _lock) =
        crate::Archive::open_primary(&dir.path().join("cache"), &dir.path().join("bytes")).unwrap();
    let object = question("sessions");
    let expected = run(
        &object,
        "confirmed",
        7,
        ChangePosition {
            generation: "synthetic-generation".into(),
            index: 5,
            archive: 4,
        },
    );
    let completed = outcome(&expected, 5);
    let mut conn = crate::archive::open_write(&archive.index_db(), 1024).unwrap();
    let tx = conn.transaction().unwrap();
    apply_run(&tx, &expected, 8, 0).unwrap();
    apply_run_outcome(&tx, &completed).unwrap();
    tx.commit().unwrap();
    drop(conn);
    let writer = crate::Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    assert_eq!(
        writer.begin_query_run(expected).unwrap(),
        QueryCustody::Canonical
    );
    assert_eq!(
        writer
            .finish_query_run(completed)
            .unwrap()
            .recv()
            .unwrap()
            .unwrap(),
        QueryCustody::Canonical
    );
    assert!(producer.writes().unwrap().is_empty());
}

#[test]
fn confirmed_outcome_first_rejects_another_begin_intent_before_primary_spooling() {
    let dir = tempfile::tempdir().unwrap();
    let producer = Arc::new(PendingProducer(dir.path().join("named.json")));
    let (archive, _lock) =
        crate::Archive::open_primary(&dir.path().join("cache"), &dir.path().join("bytes")).unwrap();
    let object = question("sessions");
    let original = run(
        &object,
        "outcome-first-pending",
        7,
        ChangePosition {
            generation: "synthetic-generation".into(),
            index: 5,
            archive: 4,
        },
    );
    let conn = crate::archive::open_write(&archive.index_db(), 1024).unwrap();
    apply_run_outcome(&conn, &outcome(&original, 1)).unwrap();
    drop(conn);
    let writer = crate::Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    let mut changed = original.clone();
    changed.machine = "another-actual-accepting-machine".into();
    assert!(matches!(
        writer.begin_query_run(changed),
        Err(StoreError::Conflict(_))
    ));
    assert!(producer.writes().unwrap().is_empty());
    assert_eq!(
        writer.begin_query_run(original).unwrap(),
        QueryCustody::Queued
    );
    assert_eq!(producer.writes().unwrap().len(), 1);
}
