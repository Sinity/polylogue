use super::*;
fn metadata(index:i64)->Metadata {Metadata{archive:"synthetic".into(),position:ChangePosition{generation:"generation".into(),index,archive:index},written_at:100+index}}
fn session(id:&str, model:&str, n:u64)->Session {
    Session{id:id.into(),native_id:id.split_once(':').unwrap().1.into(),prefix:String::new(),title:"A neutral title".into(),origin:"codex".into(),repo:Some("project".into()),last_activity:Some(10),first_prompt:Some("A neutral prompt".into()),values:BTreeMap::from([("model".into(),BTreeMap::from([(model.into(),n)]))])}
}
fn writer(path:&Path)->CatalogWriter {CatalogWriter::build(path,&metadata(1),|writer|writer.put_catalog(&query_language::catalog::Catalog::new("sessions"))).unwrap()}
#[test]
fn replacement_subtracts_only_its_contributions_and_commits_the_position() {
    let directory=tempfile::tempdir().unwrap();let path=directory.path().join("catalog");let mut writer=writer(&path);
    writer.update(&metadata(2),|writer|{writer.put_session(&session("codex:aaa","model-a",3))?;writer.put_session(&session("codex:aab","model-a",2))}).unwrap();
    writer.update(&metadata(3),|writer|writer.put_session(&session("codex:aaa","model-b",1))).unwrap();
    let reader=CatalogReader::open(&path,"synthetic").unwrap().unwrap();
    assert_eq!(reader.metadata.position,metadata(3).position);
    assert_eq!(reader.values("model","").unwrap().iter().map(|v|(v.value.as_str(),v.count)).collect::<Vec<_>>(),vec![("model-a",2),("model-b",1)]);
    assert_eq!(reader.session_details("codex:aaa").unwrap().unwrap().first_prompt.as_deref(),Some("A neutral prompt"));
    drop(reader);
    writer.update(&metadata(4),|writer|writer.remove_session("codex:aab")).unwrap();
    assert_eq!(CatalogReader::open(&path,"synthetic").unwrap().unwrap().values("model","").unwrap()[0].value,"model-b");
}
#[test]
fn a_failed_publication_rolls_back_values_and_position() {
    let directory=tempfile::tempdir().unwrap();let path=directory.path().join("catalog");let mut writer=writer(&path);
    assert!(writer.update(&metadata(2),|writer|{writer.put_session(&session("codex:abc","temporary",1))?;Err(StoreError::Invalid("synthetic failure".into()))}).is_err());
    let reader=CatalogReader::open(&path,"synthetic").unwrap().unwrap();
    assert_eq!(reader.metadata.position,metadata(1).position);assert!(reader.values("model","").unwrap().is_empty());
}
#[test]
fn prefix_updates_include_neighbors_unicode_and_removed_sessions() {
    let directory=tempfile::tempdir().unwrap();let path=directory.path().join("catalog");let mut writer=writer(&path);
    writer.update(&metadata(2),|writer|{writer.put_session(&session("codex:żaa","m",1))?;writer.put_session(&session("codex:żab","m",1))?;writer.put_session(&session("codex:żac","m",1))}).unwrap();
    let reader=CatalogReader::open(&path,"synthetic").unwrap().unwrap();
    let rows=reader.sessions("ż").unwrap();assert_eq!(rows.len(),3);assert!(rows.iter().all(|row|row.prefix==row.id));drop(reader);
    writer.update(&metadata(3),|writer|{writer.remove_session("codex:żaa")?;writer.remove_session("codex:żab")}).unwrap();
    let reader=CatalogReader::open(&path,"synthetic").unwrap().unwrap();assert_eq!(reader.sessions("ż").unwrap()[0].prefix,"codex:ż");
    assert_eq!(reader.sessions("ntrl").unwrap().len(),1,"fuzzy title fallback");
}
#[test]
fn read_only_snapshot_never_writes_and_typed_items_use_exact_identity() {
    use std::os::unix::fs::PermissionsExt;
    let directory=tempfile::tempdir().unwrap();let path=directory.path().join("catalog");let mut writer=writer(&path);
    let schema=query_language::catalog::OutputSchema{unit:Some("messages".into()),fields:vec![],main_time:None};
    writer.update(&metadata(2),|writer|{
        for name in ["one","one-more"] {writer.put_saved(&Saved{family:name.into(),revision:format!("{name}-r1"),name:name.into(),text:"messages".into(),parameters:vec![],help:"A saved query".into(),schema:Some(schema.clone())})?;}
        writer.put_kept(&Kept{id:"result-id".into(),name:Some("human name".into()),unit:"messages".into(),row_count:Some(3),rows:schema.clone(),matches:None})
    }).unwrap();
    let before=fs::metadata(&path).unwrap().modified().unwrap();
    assert_eq!(fs::metadata(&path).unwrap().permissions().mode()&0o777,0o600);
    let reader=CatalogReader::open(&path,"synthetic").unwrap().unwrap();
    assert_eq!(reader.saved("one").unwrap().len(),1);assert!(reader.saved("on").unwrap().is_empty());assert_eq!(reader.saved_names("on").unwrap().len(),2);
    assert_eq!(reader.kept("result-id").unwrap().unwrap().name.as_deref(),Some("human name"));assert!(reader.kept("result").unwrap().is_none());
    assert!(reader.conn.execute("DELETE FROM session",[]).is_err());drop(reader);
    assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(),before);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(),1);assert!(CatalogReader::open(&path,"different").unwrap().is_none());
}
