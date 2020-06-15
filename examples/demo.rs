extern crate symspell;

use std::time::Instant;
use symspell::{SymSpell, UnicodeStringStrategy, Verbosity};

fn main() {
    let mut symspell: SymSpell<UnicodeStringStrategy> = SymSpell::default();

    measure("load_dictionary", || {
        symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    });

    measure("lookup", || {
        let result = symspell.lookup("roket", Verbosity::Top, 2);
        println!("{:?}", result);
    });
}

pub fn measure<F>(name: &str, mut f: F)
where
    F: FnMut() -> (),
{
    let now = Instant::now();
    f();
    let elapsed = now.elapsed();
    let elapsed_ms =
        (elapsed.as_secs() * 1_000_000 + elapsed.subsec_micros() as u64) as f64 / 1000.0;

    if elapsed_ms < 1000.0 {
        println!("{} took {} ms", name, elapsed_ms);
    } else {
        println!("{} took {} s", name, elapsed_ms as u64 as f64 / 1000.0);
    }
}
