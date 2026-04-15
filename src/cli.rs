use anyhow::{Context, Result, anyhow, bail};

#[derive(Debug, Clone, PartialEq)]
pub struct Cli {
    pub host: Option<String>,
    pub model: Option<String>,
    pub profile: String,
    pub system: Option<String>,
    pub temp: Option<f32>,
    pub tools: Option<bool>,
    pub stream: bool,
    pub verbose: bool,
    pub prompt: Option<String>,
}

pub const HELP_TEXT: &str = concat!(
    "Usage:\n",
    "  drover \"prompt\"\n",
    "  cat file.txt | drover\n",
    "  cat file.txt | drover \"what is this about?\"\n",
    "\n",
    "Options:\n",
    "  --help             Show this help text and exit\n",
    "  --host, -H <url>   Ollama host override\n",
    "  --model, -m <name> Model selection\n",
    "  --profile, -p <name>  Config profile name\n",
    "  --system, -s <prompt>  System prompt\n",
    "  --temp, -t <value> Temperature\n",
    "  --tools <true|false>  Enable or disable tool use for this run\n",
    "  --script-output    Buffer output and suppress transient callbacks\n",
    "  --verbose, -v      Show model and timing details on stderr\n",
    "  --version          Show the crate version and exit\n",
);

impl Cli {
    pub fn parse<I>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        Self::parse_from(args).context(HELP_TEXT)
    }

    fn parse_from<I>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut cli = Self {
            host: None,
            model: None,
            profile: "default".to_owned(),
            system: None,
            temp: None,
            tools: None,
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
                "--profile" | "-p" => {
                    cli.profile = next_value(&mut args, "--profile")?;
                }
                "--system" | "-s" => {
                    cli.system = Some(next_value(&mut args, "--system")?);
                }
                "--temp" | "-t" => {
                    let value = next_value(&mut args, "--temp")?;
                    cli.temp = Some(parse_temp(&value)?);
                }
                "--tools" => {
                    let value = next_value(&mut args, "--tools")?;
                    cli.tools = Some(parse_bool_flag(&value, "--tools")?);
                }
                "--script-output" => {
                    cli.stream = false;
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

fn parse_bool_flag(value: &str, flag: &str) -> Result<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => bail!("invalid value for {flag}: {value}"),
    }
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
        assert_eq!(cli.profile, "default");
        assert_eq!(cli.system, None);
        assert_eq!(cli.temp, None);
        assert_eq!(cli.tools, None);
        assert!(cli.stream);
        assert!(!cli.verbose);
    }

    #[test]
    fn parses_all_supported_flags() {
        let cli = parse(&[
            "-m",
            "llama3",
            "-p",
            "local",
            "-H",
            "http://localhost:11434",
            "-s",
            "you are a poet",
            "--script-output",
            "--tools",
            "false",
            "-v",
            "write a sonnet",
        ])
        .unwrap();

        assert_eq!(cli.host.as_deref(), Some("http://localhost:11434"));
        assert_eq!(cli.model.as_deref(), Some("llama3"));
        assert_eq!(cli.profile, "local");
        assert_eq!(cli.system.as_deref(), Some("you are a poet"));
        assert!(!cli.stream);
        assert_eq!(cli.tools, Some(false));
        assert!(cli.verbose);
        assert_eq!(cli.prompt.as_deref(), Some("write a sonnet"));
    }

    #[test]
    fn parses_temp_space_separated_syntax() {
        let cli = parse(&["-t", "0.7", "prompt"]).unwrap();

        assert_eq!(cli.temp, Some(0.7));
        assert_eq!(cli.profile, "default");
        assert_eq!(cli.prompt.as_deref(), Some("prompt"));
    }

    #[test]
    fn rejects_missing_system_value() {
        let err = parse(&["-s"]).unwrap_err();

        assert_eq!(
            err.source().unwrap().to_string(),
            "missing value for --system"
        );
    }

    #[test]
    fn parses_profile_long_flag() {
        let cli = parse(&["--profile", "cloud", "prompt"]).unwrap();

        assert_eq!(cli.profile, "cloud");
        assert_eq!(cli.prompt.as_deref(), Some("prompt"));
    }

    #[test]
    fn rejects_missing_profile_value() {
        let err = parse(&["-p"]).unwrap_err();

        assert_eq!(
            err.source().unwrap().to_string(),
            "missing value for --profile"
        );
    }

    #[test]
    fn rejects_duplicate_positionals() {
        let err = parse(&["first", "second"]).unwrap_err();

        assert_eq!(
            err.source().unwrap().to_string(),
            "unexpected extra positional argument"
        );
    }

    #[test]
    fn rejects_unknown_flag() {
        let err = parse(&["--bogus"]).unwrap_err();

        assert_eq!(err.source().unwrap().to_string(), "unknown flag: --bogus");
    }

    #[test]
    fn rejects_invalid_temp() {
        let err = parse(&["-t", "hot"]).unwrap_err();

        assert_eq!(
            err.source().unwrap().to_string(),
            "invalid value for --temp: hot"
        );
    }

    #[test]
    fn parses_tools_flag_true() {
        let cli = parse(&["--tools", "true", "prompt"]).unwrap();

        assert_eq!(cli.tools, Some(true));
        assert_eq!(cli.prompt.as_deref(), Some("prompt"));
    }

    #[test]
    fn rejects_missing_tools_value() {
        let err = parse(&["--tools"]).unwrap_err();

        assert_eq!(
            err.source().unwrap().to_string(),
            "missing value for --tools"
        );
    }

    #[test]
    fn rejects_invalid_tools_value() {
        let err = parse(&["--tools", "maybe"]).unwrap_err();

        assert_eq!(
            err.source().unwrap().to_string(),
            "invalid value for --tools: maybe"
        );
    }

    #[test]
    fn rejects_legacy_no_tools_flag() {
        let err = parse(&["--no-tools"]).unwrap_err();

        assert_eq!(
            err.source().unwrap().to_string(),
            "unknown flag: --no-tools"
        );
    }
}
