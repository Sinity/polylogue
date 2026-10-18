//! Native selected bodies exercise the disposable scratch transaction lifetime.
use super::*;
use polylogue_model::{MessageId, SessionId, query::TextTerm};
use polylogue_store::{Error, ReadHandle, prose_search::ProseSearch};

fn selected(world: &mut World) -> [polylogue_store::contributions::MessageView; 2] {
    let mut records = Vec::new();
    for (id, text) in [
        ("scratch-first", "sharedalpha green"),
        ("scratch-second", "sharedalpha blue"),
    ] {
        let mut record: Value = serde_json::from_slice(&simple(id, json!(text))).unwrap();
        record["promptSource"] = json!("typed");
        records.push(record.to_string() + "\n");
    }
    let source = world.input("scratch-prose.jsonl", records.concat().as_bytes());
    let imported = wait_import(world, &source, "01900000-0000-7000-8000-000000000901", "20");
    assert_eq!(imported.status.code(), Some(0), "{imported:?}");
    let receipt: Value = serde_json::from_slice(&imported.stdout).unwrap();
    world.read(&reference(), receipt["input"]["id"].as_str().unwrap(), 2);
    let mut child = world.child.take().unwrap();
    assert_eq!(unsafe { libc::kill(child.0.id() as i32, libc::SIGINT) }, 0);
    assert!(child.0.wait().unwrap().success());
    drop(child);
    let reader = ReadHandle::open(&world.archive).unwrap();
    let session = SessionId::from_stored(reference()).unwrap();
    ["scratch-first", "scratch-second"].map(|native| {
        reader
            .selected_message(&MessageId::new(session.clone(), native).unwrap())
            .unwrap()
            .unwrap()
    })
}
fn until() -> Instant {
    Instant::now() + Duration::from_secs(10)
}
fn matches(search: &ProseSearch) -> Vec<String> {
    search
        .hits(
            &[TextTerm::Word("sharedalpha".into())],
            until(),
            &mut || false,
        )
        .unwrap()
        .into_iter()
        .map(|hit| hit.message)
        .collect()
}
#[test]
fn text_index_built_in_steps_uses_the_selected_bodies() {
    let mut world = World::new();
    let [first, second] = selected(&mut world);
    let mut search = ProseSearch::new().unwrap();
    assert!(matches(&search).is_empty());
    search.add(&first, until(), &mut || false).unwrap();
    assert_eq!(matches(&search), vec![first.id().as_str().to_owned()]);
    search.add(&second, until(), &mut || false).unwrap();
    let mut expected = vec![
        first.id().as_str().to_owned(),
        second.id().as_str().to_owned(),
    ];
    expected.sort_by(|a, b| b.cmp(a));
    assert_eq!(matches(&search), expected);
    assert_eq!(search.documents(), 2);
    assert_eq!(matches(&search), expected);
}
#[test]
fn cancelled_text_index_leaves_nothing_behind() {
    let mut world = World::new();
    let [first, second] = selected(&mut world);
    let mut cancelled = ProseSearch::new().unwrap();
    cancelled.add(&first, until(), &mut || false).unwrap();
    assert!(matches!(
        cancelled.add(&second, until(), &mut || true),
        Err(Error::Cancelled)
    ));
    drop(cancelled); // Failed builders are discarded, never reported as complete.
    let mut fresh = ProseSearch::new().unwrap();
    assert_eq!(fresh.documents(), 0);
    assert!(matches(&fresh).is_empty());
    fresh.add(&second, until(), &mut || false).unwrap();
    assert!(matches!(
        fresh.hits(
            &[TextTerm::Word("sharedalpha".into())],
            until(),
            &mut || true
        ),
        Err(Error::Cancelled)
    ));
    assert_eq!(matches(&fresh), vec![second.id().as_str().to_owned()]);
    drop(fresh);
    assert!(matches(&ProseSearch::new().unwrap()).is_empty());
}
