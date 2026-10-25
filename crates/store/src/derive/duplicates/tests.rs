use super::*;

fn turns(keys: &[&str]) -> Vec<Turn> {
    keys.iter()
        .enumerate()
        .map(|(i, key)| Turn {
            prompt: (i as i64, format!("m{i}")),
            key: (*key).to_owned(),
            replies: Vec::new(),
        })
        .collect()
}

#[test]
fn a_prompt_repeated_in_either_route_is_never_paired() {
    let captured = turns(&["plan it", "continue", "continue", "thanks"]);
    let exported = turns(&["plan it", "continue", "thanks"]);
    assert_eq!(paired(&captured, &exported), [(0, 0), (3, 2)]);
}

#[test]
fn pairs_keep_one_order_in_both_routes() {
    let captured = turns(&["a", "b", "c", "d"]);
    let exported = turns(&["b", "a", "c", "", "d"]);
    assert_eq!(paired(&captured, &exported), [(1, 0), (2, 2), (3, 4)]);
}

#[test]
fn empty_prompts_are_never_paired() {
    assert!(paired(&turns(&[""]), &turns(&[""])).is_empty());
}
