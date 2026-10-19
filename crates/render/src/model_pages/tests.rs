use super::*;
use crate::tests::conversation;

fn all_pages(budget: usize) -> Vec<Page> {
    let ms = conversation();
    let mut pages = vec![];
    let mut c: Option<String> = None;
    loop {
        let p = page(&ms, budget, c.as_deref());
        c = p.next.clone();
        pages.push(p);
        if c.is_none() {
            break;
        }
        assert!(pages.len() < 10_000, "pages must make progress");
    }
    pages
}

#[test]
fn pages_cover_the_whole_session_once_at_any_budget() {
    let whole = all_pages(1_000_000);
    assert_eq!(whole.len(), 1);
    let full = whole[0].text.clone();
    assert!(full.contains("→ Bash: cargo test"));
    assert!(full.contains("← error (exit 101)"));
    assert!(
        !full.contains("sk-ant-api03-AAAA"),
        "masked before measuring"
    );
    for budget in [1, 2, 5, 13, 40] {
        let joined: String = all_pages(budget).into_iter().map(|p| p.text).collect();
        assert_eq!(joined, full, "budget {budget}");
    }
}

#[test]
fn a_page_stays_within_its_budget() {
    for p in all_pages(10) {
        assert!(p.text.chars().count() <= 40, "{:?}", p.text);
        assert!(!p.text.is_empty());
    }
}
