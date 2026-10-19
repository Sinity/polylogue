#[test]
fn the_command_tree_is_well_formed() {
    super::cli::command().debug_assert();
}
