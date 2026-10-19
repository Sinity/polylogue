use clap::CommandFactory;

#[test]
fn the_command_tree_is_well_formed() {
    super::cli::Args::command().debug_assert();
}

#[test]
fn every_command_is_a_word_queries_refuse() {
    // A query starting with a command's name, or a near miss of one, is
    // refused with a suggestion; that needs every command in the list.
    for c in super::cli::Args::command().get_subcommands() {
        assert!(
            polylogue_query::COMMAND_WORDS.contains(&c.get_name()),
            "{} is missing from COMMAND_WORDS",
            c.get_name()
        );
    }
}
