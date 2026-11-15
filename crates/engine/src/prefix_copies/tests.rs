use super::copy_point;

#[test]
fn an_identical_chain_waits_for_its_source_to_be_read() {
    assert_eq!(copy_point(100, 100, 100, false), None);
    assert_eq!(copy_point(100, 100, 100, true), None);
}

#[test]
fn a_prefix_is_read_to_its_end_unless_its_end_ends_a_record() {
    // A growing file's older chain: its incomplete last record is the
    // newer chain's too.
    assert_eq!(copy_point(100, 100, 250, false), Some((100, true)));
    // An import's last record ends with its bytes, which the newer chain
    // continues: the older chain parses its last bytes itself.
    assert_eq!(copy_point(100, 100, 250, true), Some((99, false)));
}

#[test]
fn a_diverging_chain_continues_after_the_bytes_it_shares() {
    assert_eq!(copy_point(60, 100, 250, false), Some((60, false)));
    assert_eq!(copy_point(60, 100, 80, true), Some((60, false)));
    // A newer chain that is a prefix of the older one (a truncated file).
    assert_eq!(copy_point(80, 100, 80, false), Some((80, false)));
}
