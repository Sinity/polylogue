use std::{collections::BTreeSet, env, fs, path::PathBuf};

// The parser and discovery read the same grammar, including case-sensitive words.
fn words(text: &str) -> Vec<&str> {
    text.split('"')
        .enumerate()
        .filter(|(i, word)| {
            i % 2 == 1
                && word.chars().any(|c| c.is_ascii_alphabetic())
                && word
                    .chars()
                    .all(|c| c.is_ascii_alphabetic() || c == '_' || c == ':')
        })
        .map(|(_, word)| word)
        .collect()
}

fn references(text: &str) -> impl Iterator<Item = &str> {
    text.split('"').step_by(2).flat_map(|text| {
        text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|word| !word.is_empty())
    })
}

fn leading_words(name: &str, rules: &std::collections::BTreeMap<String, String>) -> Vec<String> {
    let Some(rule) = rules.get(name) else {
        return Vec::new();
    };
    let words = words(rule);
    if words.is_empty() {
        return references(rule)
            .next()
            .map_or_else(Vec::new, |name| leading_words(name, rules));
    }
    if rule.contains('|') {
        words.into_iter().map(str::to_owned).collect()
    } else {
        vec![words.join(" ")]
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/query.pest");
    let grammar = fs::read_to_string("src/query.pest").expect("query grammar");
    let mut rules = std::collections::BTreeMap::<String, String>::new();
    let mut current = None;
    for line in grammar
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
    {
        if let Some((name, body)) = line.split_once('=')
            && !name.trim().is_empty()
            && name
                .trim()
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            current = Some(name.trim().to_owned());
            rules.insert(name.trim().to_owned(), body.to_owned());
        } else if let Some(name) = &current {
            rules.get_mut(name).expect("current rule").push_str(line);
        }
    }
    let keywords: BTreeSet<_> = rules.values().flat_map(|rule| words(rule)).collect();
    let stages: Vec<_> = references(&rules["stage"])
        .flat_map(|name| leading_words(name, &rules))
        .collect();
    let mut aggregates = words(&rules["agg_name"]);
    aggregates.extend(words(&rules["agg_fn"]));
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
