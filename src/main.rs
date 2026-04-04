mod cli;
mod config;
mod ollama;
mod run;

use cli::Cli;

fn main() {
    let args = std::env::args().skip(1);

    if let Err(err) = Cli::parse(args).and_then(|cli| cli.run()) {
        #[cfg(debug_assertions)]
        {
            eprintln!("{:?}", err);
        }

        #[cfg(not(debug_assertions))]
        {
            eprintln!(
                "{}",
                err.chain()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }

        std::process::exit(1);
    }
}
