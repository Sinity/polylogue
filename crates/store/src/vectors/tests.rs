use super::*;

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

#[test]
fn stores_scans_and_survives_reopening() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    let mut v = Vectors::open(&archive).unwrap();
    let key = "local/test/3/document/unit";
    v.put(
        key,
        &[
            Vector {
                text_hash: vec![1],
                chunk: 0,
                values: unit(&[1.0, 0.0, 0.0]),
            },
            Vector {
                text_hash: vec![2],
                chunk: 0,
                values: unit(&[0.0, 1.0, 0.0]),
            },
            // A second chunk of text 2, nearer the query: the text's
            // best chunk counts.
            Vector {
                text_hash: vec![2],
                chunk: 1,
                values: unit(&[0.9, 0.1, 0.0]),
            },
        ],
    )
    .unwrap();
    v.fail(key, &[3], "too long").unwrap();
    drop(v);
    let v = Vectors::open_read(&archive).unwrap();
    let near = v.nearest(key, &unit(&[1.0, 0.05, 0.0]), 2).unwrap();
    assert_eq!(near.len(), 2);
    assert_eq!(near[0].0, vec![1]);
    assert!(near[0].1 >= near[1].1);
    assert_eq!(v.done(key).unwrap().len(), 3);
    let s = v.status().unwrap();
    assert_eq!(s.keys[0].texts, 2);
    assert_eq!(s.keys[0].vectors, 3);
    assert_eq!(s.keys[0].failures, 1);
    // Another key's vectors never answer this one.
    assert!(v.nearest("other", &[1.0, 0.0, 0.0], 5).unwrap().is_empty());
}
