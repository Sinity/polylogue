use super::*;
use polylogue_model::read::Shown;

fn sums(costs: &[u64]) -> Arc<RunningSums> {
    let mut sums = vec![0u64];
    for &cost in costs {
        sums.push(sums.last().unwrap().saturating_add(cost));
    }
    Arc::new(RunningSums {
        shown: costs
            .iter()
            .enumerate()
            .map(|(entry, &cost)| Shown {
                segment: 0,
                entry,
                cost,
            })
            .collect(),
        sums,
    })
}

#[test]
fn composed_ranges_page_like_the_flat_reading_without_collecting_it() {
    let mut index = ShownEntries {
        runs: vec![],
        len: 0,
        tokens: 0,
    };
    let first = sums(&[90, 0, 3, 8, 0, 2, 90]);
    let second = sums(&[70, 5, 0, 4, 70]);
    index.push(0, first, 1..6);
    index.push(1, second, 1..4);
    let places = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 1),
        (1, 2),
        (1, 3),
    ];
    let flat = sums(&[0, 3, 8, 0, 2, 5, 0, 4]);
    assert_eq!(index.range(0..index.len()), places);
    assert_eq!(index.get(index.len()), None);
    for end in 0..=places.len() {
        assert_eq!(index.prefix(end), flat.sums[end]);
        for budget in 0..=30 {
            for items in 0..=10 {
                assert_eq!(
                    index.page_after(end, budget, items),
                    flat.page_after(end, budget, items),
                    "forward {end}/{budget}/{items}"
                );
                assert_eq!(
                    index.page_before(end, budget, items),
                    flat.page_before(end, budget, items),
                    "backward {end}/{budget}/{items}"
                );
            }
        }
    }
    for part in 0..3 {
        for entry in 0..8 {
            assert_eq!(
                index.partition_point(|place| place < (part, entry)),
                places.partition_point(|place| *place < (part, entry))
            );
        }
    }
}

#[test]
fn empty_ranges_do_not_create_a_page() {
    let mut index = ShownEntries {
        runs: vec![],
        len: 0,
        tokens: 0,
    };
    index.push(0, sums(&[1, 2]), 1..1);
    assert_eq!(index.len(), 0);
    assert_eq!(index.get(0), None);
    assert_eq!(index.page_after(0, 0, 0), 0);
    assert_eq!(index.page_before(0, 0, 0), 0);
}
