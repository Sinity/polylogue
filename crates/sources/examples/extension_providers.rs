//! Prints the browser extension's provider table from the origin registry.
//! `cargo xtask extension-providers` writes it to
//! `extension/src/background/providers.js`.

fn main() {
    print!(
        "{}",
        polylogue_sources::origin::extension_providers_module()
    );
}
