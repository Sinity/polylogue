use clap::CommandFactory;

#[test]
fn the_command_tree_is_well_formed() {
    super::cli::Args::command().debug_assert();
}
