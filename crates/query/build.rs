use std::{collections::BTreeSet, env, fs, path::PathBuf};

// The parser and discovery read the same grammar, including case-sensitive words.
fn words(text: &str) -> Vec<&str> {
    text.split('"')
        .enumerate()
        .filter(|(i, word)| {
            i % 2 == 1
                && !word.is_empty()
                && word
                    .chars()
                    .all(|c| c.is_ascii_alphabetic() || c == '_' || c == ':')
        })
        .map(|(_, word)| word)
        .collect()
}

fn main() {
    println!("cargo:rerun-if-changed=src/query.pest");
    let grammar = fs::read_to_string("src/query.pest").expect("query grammar");
    let rules: Vec<_> = grammar
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect();
    let keywords: BTreeSet<_> = rules.iter().flat_map(|line| words(line)).collect();
    let rule = |name: &str| -> &str {
        rules
            .iter()
            .find(|line| {
                line.split_once('=')
                    .is_some_and(|(left, _)| left.trim() == name)
            })
            .expect("grammar rule")
    };
    let mut stages = vec!["sort by", "group by"];
    for name in [
        "agg",
        "count",
        "limit",
        "offset",
        "distinct",
        "matches",
        "measure",
        "combination",
    ] {
        stages.extend(words(rule(name)));
    }
    stages.extend(words(rule("kw_where")));
    let mut aggregates = words(rule("agg_name"));
    aggregates.extend(words(rule("count")));
    let output = format!(
        "pub const KEYWORDS: &[&str] = &{:?};\npub const STAGES: &[&str] = &{:?};\npub const AGGREGATES: &[&str] = &{:?};\n",
        keywords.into_iter().collect::<Vec<_>>(),
        stages,
        aggregates,
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").expect("build output")).join("syntax.rs"),
        output,
    )
    .expect("syntax declarations");
}
