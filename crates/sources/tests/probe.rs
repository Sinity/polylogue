//! Prints a parser's output for `PROBE_ORIGIN`, `PROBE_PATH` and the bytes
//! in `PROBE_INPUT` (a file). Used while porting knowledge from the Python
//! predecessor's tests; ignored in normal runs.
mod support;

#[test]
#[ignore]
fn probe() {
    let origin = std::env::var("PROBE_ORIGIN").unwrap();
    let path = std::env::var("PROBE_PATH").unwrap();
    let input = std::fs::read(std::env::var("PROBE_INPUT").unwrap()).unwrap();
    println!("{:#}", support::parse(&origin, &path, &input));
}
