use super::*;
use crate::{Archive,ReadPool,StoreError,completion::{CatalogReader,CatalogWriter,Metadata}};
use std::sync::Arc;
fn build(pointer:&CatalogPointer,generation:&str,value:&str) {
    let metadata=Metadata{archive:"synthetic".into(),position:polylogue_model::ChangePosition{generation:generation.into(),index:1,archive:1},written_at:100};
    CatalogWriter::build(&pointer.generation_path(generation),&metadata,|writer|{
        writer.put_catalog(&query_language::catalog::Catalog::new("sessions"))?;
        writer.put_values("synthetic",&std::collections::BTreeMap::from([("repo".into(),std::collections::BTreeMap::from([(value.into(),1)]))]))
    }).unwrap();
}
#[test]
fn revoked_pointer_rejects_open_snapshots_and_a_late_old_generation_publisher() {
    let directory=tempfile::tempdir().unwrap();let root=directory.path().join("archive");
    let pointer=Arc::new(CatalogPointer::new(directory.path().join("catalog")));
    let invalidate=Arc::clone(&pointer);
    let _registration=Archive::register_scope_invalidator(&root,move||invalidate.invalidate()).unwrap();
    let (archive,_lock)=Archive::open_or_create(&root).unwrap();
    let reads=ReadPool::open(&archive).unwrap();let generation=reads.reader().unwrap().change_position().unwrap().generation;
    let old=archive.scope_token().unwrap();build(&pointer,&generation,"old-value");pointer.activate(&generation,&old).unwrap();
    let snapshot=CatalogReader::open(&pointer.path,"synthetic").unwrap().unwrap();
    assert_eq!(snapshot.values("repo","").unwrap()[0].value,"old-value");
    let state=crate::read::scope::state(&root);
    {let mut reset=state.begin_reset().unwrap();reset.invalidate().unwrap();}
    assert!(matches!(snapshot.check(),Err(StoreError::ScopeChanged)));
    assert!(CatalogReader::open(&pointer.path,"synthetic").unwrap().is_none());
    assert!(matches!(pointer.activate(&generation,&old),Err(StoreError::ScopeChanged)));
    state.finish_reset();
    build(&pointer,"new-generation","new-value");pointer.activate("new-generation",&archive.scope_token().unwrap()).unwrap();
    let current=CatalogReader::open(&pointer.path,"synthetic").unwrap().unwrap();
    assert_eq!(current.values("repo","").unwrap()[0].value,"new-value");
    assert!(matches!(snapshot.check(),Err(StoreError::ScopeChanged)));
}
#[test]
fn ordinary_reopen_invalidates_the_pointer_before_a_new_publisher_runs() {
    let directory=tempfile::tempdir().unwrap();let root=directory.path().join("archive");
    let pointer=Arc::new(CatalogPointer::new(directory.path().join("catalog")));
    let invalidator=Arc::clone(&pointer);
    let _registration=Archive::register_scope_invalidator(&root,move||invalidator.invalidate()).unwrap();
    let (archive,lock)=Archive::open_or_create(&root).unwrap();
    let token=archive.scope_token().unwrap();build(&pointer,"first","cached");pointer.activate("first",&token).unwrap();
    drop(lock);drop(archive);
    let (_archive,_lock)=Archive::open_or_create(&root).unwrap();
    assert!(!pointer.path.exists());assert!(matches!(token.check(),Err(StoreError::ScopeChanged)));
}
