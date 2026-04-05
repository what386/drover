use anyhow::{Context, Result, anyhow, bail};

#[derive(Debug, Clone, PartialEq)]
pub struct Cli {
    pub host: Option<String>,
    pub model: Option<String>,
    pub system: Option<String>,
    pub temp: Option<f32>,
    pub no_tools: bool,
    pub stream: bool,
    pub verbose: bool,
    pub prompt: Option<String>,
}

impl Cli {
    pub fn parse<I>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        Self::parse_from(args).with_context(Self::invalid_input_help)
    }

    pub fn invalid_input_help() -> String {
        format!("invalid command line input\n\n{}", Self::help_text())
    }

    pub fn help_text() -> &'static str {
        concat!(
            "Usage:\n",
            "  drover \"prompt\"\n",
            "  cat file.txt | drover\n",
            "  cat file.txt | drover \"what is this about?\"\n",
            "\n",
            "Options:\n",
            "  --help             Show this help text and exit\n",
            "  --host, -H <url>   Ollama host override\n",
            "  --model, -m <name> Model selection\n",
            "  --system, -s <prompt>  System prompt\n",
            "  --temp, -t <value> Temperature\n",
            "  --no-tools         Disable tool use for this run\n",
            "  --script-output    Buffer output and suppress transient callbacks\n",
            "  --verbose, -v      Show model and timing details on stderr\n",
            "  --version          Show the crate version and exit\n",
        )
    }

    fn parse_from<I>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut cli = Self {
            host: None,
            model: None,
            system: None,
            temp: None,
            no_tools: false,
            stream: true,
            verbose: false,
            prompt: None,
        };

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--host" | "-H" => {
                    cli.host = Some(next_value(&mut args, "--host")?);
                }
                "--model" | "-m" => {
                    cli.model = Some(next_value(&mut args, "--model")?);
                }
                "--system" | "-s" => {
                    cli.system = Some(next_value(&mut args, "--system")?);
                }
                "--temp" | "-t" => {
                    let value = next_value(&mut args, "--temp")?;
                    cli.temp = Some(parse_temp(&value)?);
                }
                "--script-output" => {
                    cli.stream = false;
                }
                "--no-tools" => {
                    cli.no_tools = true;
                }
                "--verbose" | "-v" => {
                    cli.verbose = true;
                }
                _ if arg.starts_with("--") => {
                    return Err(anyhow!("unknown flag: {arg}"));
                }
                _ => {
                    if cli.prompt.replace(arg).is_some() {
                        bail!("unexpected extra positional argument");
                    }
                }
            }
        }

        Ok(cli)
    }
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| anyhow!("missing value for {flag}"))
}

fn parse_temp(value: &str) -> Result<f32> {
    value
        .parse::<f32>()
        .context(format!("invalid value for --temp: {value}"))
}

#[cfg(test)]
mod tests {
    use super::Cli;

    fn parse(args: &[&str]) -> anyhow::Result<Cli> {
        Cli::parse(args.iter().map(|arg| (*arg).to_owned()))
    }

    #[test]
    fn parses_prompt_only() {
        let cli = parse(&["write a haiku"]).unwrap();

        assert_eq!(cli.prompt.as_deref(), Some("write a haiku"));
        assert_eq!(cli.host, None);
        assert_eq!(cli.model, None);
        assert_eq!(cli.system, None);
        assert_eq!(cli.temp, None);
        assert!(!cli.no_tools);
        assert!(cli.stream);
        assert!(!cli.verbose);
    }

    #[test]
    fn parses_stdin_style_invocation_without_prompt() {
        let cli = parse(&[]).unwrap();

        assert_eq!(cli.prompt, None);
        assert!(cli.stream);
    }

    #[test]
    fn parses_all_supported_flags() {
        let cli = parse(&[
            "-m",
            "llama3",
            "-H",
            "http://localhost:11434",
            "-s",
            "you are a poet",
            "--script-output",
            "--no-tools",
            "-v",
            "write a sonnet",
        ])
        .unwrap();

        assert_eq!(cli.host.as_deref(), Some("http://localhost:11434"));
        assert_eq!(cli.model.as_deref(), Some("llama3"));
        assert_eq!(cli.system.as_deref(), Some("you are a poet"));
        assert!(!cli.stream);
        assert!(cli.no_tools);
        assert!(cli.verbose);
        assert_eq!(cli.prompt.as_deref(), Some("write a sonnet"));
    }

    #[test]
    fn parses_temp_space_separated_syntax() {
        let cli = parse(&["-t", "0.7", "prompt"]).unwrap();

        assert_eq!(cli.temp, Some(0.7));
        assert_eq!(cli.prompt.as_deref(), Some("prompt"));
    }

    #[test]
    fn rejects_missing_model_value() {
        let err = parse(&["-m"]).unwrap_err();

        assert_eq!(err.to_string(), Cli::invalid_input_help());
        assert_eq!(
            err.source().unwrap().to_string(),
            "missing value for --model"
        );
    }

    #[test]
    fn rejects_missing_host_value() {
        let err = parse(&["-H"]).unwrap_err();

        assert_eq!(err.to_string(), Cli::invalid_input_help());
        assert_eq!(
            err.source().unwrap().to_string(),
            "missing value for --host"
        );
    }

    #[test]
    fn rejects_missing_system_value() {
        let err = parse(&["-s"]).unwrap_err();

        assert_eq!(err.to_string(), Cli::invalid_input_help());
        assert_eq!(
            err.source().unwrap().to_string(),
            "missing value for --system"
        );
    }

    #[test]
    fn rejects_duplicate_positionals() {
        let err = parse(&["first", "second"]).unwrap_err();

        assert_eq!(err.to_string(), Cli::invalid_input_help());
        assert_eq!(
            err.source().unwrap().to_string(),
            "unexpected extra positional argument"
        );
    }

    #[test]
    fn rejects_unknown_flag() {
        let err = parse(&["--bogus"]).unwrap_err();

        assert_eq!(err.to_string(), Cli::invalid_input_help());
        assert_eq!(err.source().unwrap().to_string(), "unknown flag: --bogus");
    }

    #[test]
    fn rejects_invalid_temp() {
        let err = parse(&["-t", "hot"]).unwrap_err();

        assert_eq!(err.to_string(), Cli::invalid_input_help());
        assert_eq!(
            err.source().unwrap().to_string(),
            "invalid value for --temp: hot"
        );
    }

    #[test]
    fn rejects_removed_no_stream_flag() {
        let err = parse(&["--no-stream"]).unwrap_err();

        assert_eq!(err.to_string(), Cli::invalid_input_help());
        assert_eq!(
            err.source().unwrap().to_string(),
            "unknown flag: --no-stream"
        );
    }

    #[test]
    fn help_text_includes_help_and_version_flags() {
        let help = Cli::help_text();

        assert!(help.contains("--help"));
        assert!(help.contains("--host, -H <url>"));
        assert!(help.contains("--version"));
        assert!(help.contains("--temp, -t <value>"));
        assert!(help.contains("--no-tools"));
        assert!(help.contains("--script-output"));
        assert!(help.contains("--model, -m <name>"));
        assert!(help.contains("--system, -s <prompt>"));
        assert!(help.contains("--verbose, -v"));
    }

    #[test]
    fn parses_mixed_short_and_long_flags() {
        let cli = parse(&[
            "--model",
            "llama3",
            "-H",
            "http://localhost:11434",
            "--system",
            "you are a poet",
            "-t",
            "0.7",
            "-v",
            "write a sonnet",
        ])
        .unwrap();

        assert_eq!(cli.host.as_deref(), Some("http://localhost:11434"));
        assert_eq!(cli.model.as_deref(), Some("llama3"));
        assert_eq!(cli.system.as_deref(), Some("you are a poet"));
        assert_eq!(cli.temp, Some(0.7));
        assert!(!cli.no_tools);
        assert!(cli.verbose);
        assert!(cli.stream);
    }

    #[test]
    fn parses_no_tools_flag() {
        let cli = parse(&["--no-tools", "prompt"]).unwrap();

        assert!(cli.no_tools);
        assert_eq!(cli.prompt.as_deref(), Some("prompt"));
    }
}
