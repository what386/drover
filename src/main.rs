mod cli;
mod config;
mod ollama;
mod run;
mod tools;

use cli::Cli;

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();

    if args.iter().any(|arg| arg == "--help") {
        println!("{}", Cli::help_text());
        return;
    }

    if args.iter().any(|arg| arg == "--version") {
        println!("drover v{}", env!("CARGO_PKG_VERSION"));
        return;
    }

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
