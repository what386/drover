use crate::cli::Cli;
use crate::config::Config;
use crate::ollama::{GenerateChunk, GenerateRequest, GenerateResponse, OllamaClient};
use anyhow::{Context, Result, anyhow, bail};
use std::io::{self, IsTerminal, Read, Write};
use std::time::{Duration, Instant};

impl Cli {
    pub fn run(self) -> Result<()> {
        let stdin_is_terminal = io::stdin().is_terminal();
        let prompt = self.resolve_prompt(stdin_is_terminal)?;
        let config = Config::load().context("failed to load config")?;
        let request = self.build_request(&config, prompt)?;
        let host = self.resolve_host(&config)?;
        let model = request.model.clone();
        let client = OllamaClient::new(host.clone());
        let started_at = Instant::now();

        let response = if request.stream {
            self.run_streaming(&client, &request)?
        } else {
            self.run_non_streaming(&client, &request)?
        };

        if self.verbose {
            self.print_verbose(&host, &model, &request, &response, started_at.elapsed())?;
        }

        Ok(())
    }

    fn resolve_prompt(&self, stdin_is_terminal: bool) -> Result<String> {
        let stdin = if stdin_is_terminal {
            None
        } else {
            Some(Self::read_stdin()?)
        };

        Self::compose_prompt(self.prompt.clone(), stdin)
    }

    fn read_stdin() -> Result<String> {
        let mut stdin = String::new();
        io::stdin()
            .read_to_string(&mut stdin)
            .context("failed to read stdin")?;

        if stdin.is_empty() {
            bail!("stdin was empty");
        }

        Ok(stdin)
    }

    fn compose_prompt(prompt: Option<String>, stdin: Option<String>) -> Result<String> {
        match (prompt, stdin) {
            (Some(prompt), Some(stdin)) => Ok(format!("{prompt}\n\nInput:\n{stdin}")),
            (Some(prompt), None) => Ok(prompt),
            (None, Some(stdin)) => Ok(stdin),
            (None, None) => bail!(
                "expected a prompt argument or piped stdin input\n\n{}",
                Self::help_text()
            ),
        }
    }

    fn build_request(&self, config: &Config, prompt: String) -> Result<GenerateRequest> {
        let model = self
            .model
            .clone()
            .or_else(|| config.model.clone())
            .ok_or_else(|| anyhow!("config is missing `model`"))?;

        let temp = self.temp.or(config.temp);

        Ok(GenerateRequest {
            model,
            prompt,
            system: self.system.clone(),
            temp,
            stream: self.stream,
        })
    }

    fn resolve_host(&self, config: &Config) -> Result<String> {
        self.host
            .clone()
            .or_else(|| config.host.clone())
            .ok_or_else(|| anyhow!("config is missing `host`"))
    }

    fn run_non_streaming(
        &self,
        client: &OllamaClient,
        request: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let response = client.generate(request)?;
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(response.response.as_bytes())
            .context("failed to write response to stdout")?;
        if !response.response.ends_with('\n') {
            stdout
                .write_all(b"\n")
                .context("failed to write newline to stdout")?;
        }
        stdout.flush().context("failed to flush stdout")?;
        Ok(response)
    }

    fn run_streaming(
        &self,
        client: &OllamaClient,
        request: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let mut stdout = io::stdout().lock();
        let response = client.generate_streaming(request, |chunk: &GenerateChunk| {
            if !chunk.response.is_empty() {
                stdout
                    .write_all(chunk.response.as_bytes())
                    .context("failed to write response chunk to stdout")?;
                stdout.flush().context("failed to flush stdout")?;
            }
            Ok(())
        })?;

        if !response.response.ends_with('\n') {
            stdout
                .write_all(b"\n")
                .context("failed to write newline to stdout")?;
            stdout.flush().context("failed to flush stdout")?;
        }

        Ok(response)
    }

    fn print_verbose(
        &self,
        host: &str,
        model: &str,
        request: &GenerateRequest,
        response: &GenerateResponse,
        elapsed: Duration,
    ) -> Result<()> {
        let mut stderr = io::stderr().lock();
        writeln!(stderr, "host: {host}").context("failed to write verbose output")?;
        writeln!(stderr, "model: {model}").context("failed to write verbose output")?;
        writeln!(stderr, "stream: {}", request.stream).context("failed to write verbose output")?;
        if let Some(temp) = request.temp {
            writeln!(stderr, "temp: {temp}").context("failed to write verbose output")?;
        }
        writeln!(stderr, "elapsed_ms: {}", elapsed.as_millis())
            .context("failed to write verbose output")?;
        if let Some(total_duration) = response.total_duration {
            writeln!(stderr, "total_duration_ns: {total_duration}")
                .context("failed to write verbose output")?;
        }
        if let Some(load_duration) = response.load_duration {
            writeln!(stderr, "load_duration_ns: {load_duration}")
                .context("failed to write verbose output")?;
        }
        if let Some(prompt_eval_count) = response.prompt_eval_count {
            writeln!(stderr, "prompt_eval_count: {prompt_eval_count}")
                .context("failed to write verbose output")?;
        }
        if let Some(eval_count) = response.eval_count {
            writeln!(stderr, "eval_count: {eval_count}")
                .context("failed to write verbose output")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::Cli;
    use crate::config::Config;

    #[test]
    fn resolve_prompt_prefers_cli_prompt() {
        let cli = Cli {
            host: None,
            model: None,
            system: None,
            temp: None,
            stream: true,
            verbose: false,
            prompt: Some("prompt".to_owned()),
        };

        assert_eq!(Cli::compose_prompt(cli.prompt.clone(), None).unwrap(), "prompt");
    }

    #[test]
    fn resolve_prompt_requires_input_when_stdin_is_terminal() {
        let err = Cli::compose_prompt(None, None).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "expected a prompt argument or piped stdin input\n\n{}",
                Cli::help_text()
            )
        );
    }

    #[test]
    fn compose_prompt_uses_stdin_only_when_no_prompt_is_provided() {
        let prompt = Cli::compose_prompt(None, Some("file contents".to_owned())).unwrap();

        assert_eq!(prompt, "file contents");
    }

    #[test]
    fn compose_prompt_combines_prompt_and_stdin() {
        let prompt = Cli::compose_prompt(
            Some("what is this project about?".to_owned()),
            Some("README body".to_owned()),
        )
        .unwrap();

        assert_eq!(prompt, "what is this project about?\n\nInput:\nREADME body");
    }

    #[test]
    fn build_request_prefers_cli_values_over_config() {
        let cli = Cli {
            host: Some("http://override:11434".to_owned()),
            model: Some("cli-model".to_owned()),
            system: Some("sys".to_owned()),
            temp: Some(0.1),
            stream: false,
            verbose: false,
            prompt: Some("prompt".to_owned()),
        };
        let config = Config {
            model: Some("cfg-model".to_owned()),
            host: Some("http://localhost:11434".to_owned()),
            temp: Some(0.7),
        };

        let request = cli.build_request(&config, "prompt".to_owned()).unwrap();

        assert_eq!(request.model, "cli-model");
        assert_eq!(request.system.as_deref(), Some("sys"));
        assert_eq!(request.temp, Some(0.1));
        assert!(!request.stream);
        assert_eq!(cli.resolve_host(&config).unwrap(), "http://override:11434");
    }

    #[test]
    fn build_request_falls_back_to_config() {
        let cli = Cli {
            host: None,
            model: None,
            system: None,
            temp: None,
            stream: true,
            verbose: false,
            prompt: None,
        };
        let config = Config::default();

        let request = cli.build_request(&config, "prompt".to_owned()).unwrap();

        assert_eq!(request.model, "llama3");
        assert_eq!(request.temp, Some(0.7));
        assert!(request.stream);
        assert_eq!(cli.resolve_host(&config).unwrap(), "http://localhost:11434");
    }
}
