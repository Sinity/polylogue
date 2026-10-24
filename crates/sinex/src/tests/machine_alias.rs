use super::*;
use polylogue_store::sinex_rows::{Published, Unit};
use polylogue_store::writer::{Batch, InFlight, InputKey, Observation};
use polylogue_store::{Change, Writer};

fn payload(source: &Published, unit: &Unit) -> Vec<u8> {
    let mut batch = source.read(unit).unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut batch.file, &mut bytes).unwrap();
    bytes
}

pub(super) async fn rename(sinex: &mut Sinex, root: &Path) {
    let path = root.join("machine-alias-source");
    let (source, lock) = Archive::open_or_create(&path).unwrap();
    let writer = Writer::start(&source).unwrap();
    let observe = |host: &str, at, change| {
        writer
            .send(Batch {
                input: InputKey {
                    id: "synthetic-machine-input".into(),
                    format: "synthetic-text".into(),
                    route_rank: 0,
                },
                change,
                observation: Some(Observation {
                    machine: "synthetic-machine".into(),
                    host: host.into(),
                    path: "synthetic".into(),
                    facts: None,
                    seen_at_ms: at,
                }),
                continues: false,
                piece: None,
                blobs: vec![],
                length: 0,
                parse: vec![],
                release: InFlight::default(),
            })
            .unwrap();
        assert!(writer.flush().unwrap().errors.is_empty());
    };
    observe("host-a", 900, Change::First);
    let published = Published::open(&path).unwrap();
    let id = published.archive_id().unwrap();
    let unit = inventory_units(&published)
        .into_iter()
        .find(|u| u.kind == "observation")
        .unwrap();
    let before = payload(&published, &unit);
    let hash = blake3::hash(&before);
    drop(published);
    let first = sinex.publish(&path).await.unwrap();
    assert_eq!((first.units, first.held), (2, 0));
    observe("host-b", 1, Change::Unchanged);
    let published = Published::open(&path).unwrap();
    assert_eq!(payload(&published, &unit), before);
    assert_eq!(blake3::hash(&payload(&published, &unit)), hash);
    drop(published);
    let next = sinex.publish(&path).await.unwrap();
    assert_eq!((next.units, next.held), (1, 2));
    drop(writer);
    drop(lock);
    let target = root.join("machine-alias-restored");
    let (restored, lock) = Archive::open_or_create(&target).unwrap();
    let deadline = Instant::now();
    loop {
        if sinex.restore(&restored, &id).await.unwrap().units == 3 {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(lock);
    let (restored, _lock) = Archive::open_or_create(&target).unwrap();
    let published = Published::open(&target).unwrap();
    assert_eq!(payload(&published, &unit), before);
    assert_eq!(blake3::hash(&payload(&published, &unit)), hash);
    let db = polylogue_store::archive::open_read(&restored.archive_db()).unwrap();
    let hosts: Vec<String> = db
        .prepare("SELECT host FROM machine WHERE id='synthetic-machine' ORDER BY host")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(hosts, ["host-a", "host-b"]);
    assert_eq!(sinex.publish(&target).await.unwrap().units, 0);
}
