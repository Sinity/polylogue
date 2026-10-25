use super::*;
use polylogue_render::{
    View,
    model_pages::{Pager, Position},
};

fn key() -> Key {
    Key {
        binding: "selection-view".into(),
        content: "content".into(),
        generation: "generation".into(),
        index: (1, 2),
    }
}
fn checkpoint(message: usize) -> Checkpoint {
    let source: polylogue_model::Message = serde_json::from_value(serde_json::json!({"id":"claude-code:m", "session":"claude-code:s", "role":"user", "blocks":[{"type":"text","text":"visible synthetic content"}]})).unwrap();
    let mut pager = Pager::new(
        View::Full,
        1,
        Some(Position {
            message,
            piece: 0,
            character: 0,
        }),
    );
    for _ in 0..=message {
        if pager.push(source.clone()).unwrap().is_break() {
            break;
        }
    }
    pager.checkpoint().unwrap()
}

#[test]
fn matching_private_checkpoints_are_bounded_and_content_bound() {
    let cache = Cache::default();
    let epoch = cache.epoch();
    let key = key();
    assert!(cache.get(&key, epoch, 0).is_none());
    for message in 0..=ENTRIES {
        cache.put(key.clone(), epoch, checkpoint(message));
    }
    assert_eq!(cache.state().entries.len(), ENTRIES);
    assert!(
        cache.get(&key, epoch, 0).is_none(),
        "oldest entry must be evicted"
    );
    assert_eq!(cache.get(&key, epoch, 3).unwrap().message(), 3);
    assert_eq!(
        cache.get(&key, epoch, usize::MAX).unwrap().message(),
        ENTRIES
    );
    let mut other = key.clone();
    other.content = "changed".into();
    assert!(cache.get(&other, epoch, 3).is_none());
    other = key.clone();
    other.binding = "other view".into();
    assert!(cache.get(&other, epoch, 3).is_none());
    // No commit sequence participates: unrelated ingestion leaves a hit warm.
    assert!(cache.get(&key, epoch, 3).is_some());
    other = key.clone();
    other.index = (1, 3);
    assert!(cache.get(&other, epoch, 3).is_none());
    assert!(cache.state().entries.is_empty());
    cache.put(key, epoch, checkpoint(0));
    assert!(
        cache.state().entries.is_empty(),
        "a cleared key cannot be stored again"
    );
}

#[test]
fn mutation_epochs_refuse_inflight_private_state_before_and_after_writes() {
    let cache = Cache::default();
    let key = key();
    let original = cache.epoch();
    cache.get(&key, original, 0);
    cache.put(key.clone(), original, checkpoint(0));
    let during;
    {
        let _mutation = cache.mutation();
        during = cache.epoch();
        assert!(cache.state().entries.is_empty());
        cache.put(key.clone(), original, checkpoint(0));
        assert!(cache.state().entries.is_empty());
        cache.put(key.clone(), during, checkpoint(0));
        assert_eq!(cache.state().entries.len(), 1);
    }
    assert!(cache.state().entries.is_empty());
    cache.put(key.clone(), during, checkpoint(0));
    assert!(cache.state().entries.is_empty());
    let after = cache.epoch();
    cache.put(key.clone(), after, checkpoint(0));
    assert!(cache.get(&key, after, 0).is_some());
    cache.clear();
    assert!(cache.state().entries.is_empty());
}

#[test]
fn a_new_index_generation_drops_cached_state_and_refuses_entries_from_the_old_one() {
    let cache = Cache::default();
    let key = key();
    let epoch = cache.epoch();
    cache.get(&key, epoch, 0);
    cache.put(key.clone(), epoch, checkpoint(0));
    let mut rebuilt = key.clone();
    rebuilt.generation = "rebuilt".into();
    assert!(cache.get(&rebuilt, epoch, 0).is_none());
    assert!(cache.state().entries.is_empty());
    cache.put(key, epoch, checkpoint(0));
    assert!(cache.state().entries.is_empty());
    let current = cache.epoch();
    cache.put(rebuilt.clone(), current, checkpoint(0));
    assert!(cache.get(&rebuilt, current, 0).is_some());
}
