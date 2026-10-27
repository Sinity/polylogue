use super::*;

#[test]
fn five_minute_clustering_uses_interval_bounds_and_absolute_gaps() {
    assert_eq!(clustered(&[]), 0);
    assert_eq!(clustered(&[0, 300_000, 600_000]), 600_000);
    assert_eq!(clustered(&[0, 360_000, 720_000]), 0);
    assert_eq!(clustered(&[200, 0, 100]), 200);
    assert_eq!(clustered(&[0, 120_000, 900_000, 1_020_000]), 240_000);
}

#[test]
fn topic_strips_known_prefixes_and_preserves_unicode() {
    assert_eq!(clean_topic("[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code. Fix the parser.",120).as_deref(),Some("Fix the parser."));
    assert_eq!(clean_topic("   ", 120), None);
    assert_eq!(clean_topic("ééé", 2).as_deref(), Some("éé"));
}
