use crate::cli::Cli;
use crate::config::Config;
use crate::ollama::{GenerateRequest, GenerateResponse, OllamaClient, StreamCallbackAction};
use crate::tools::{
    TOOL_SYSTEM_PROMPT, ToolCall, execute_tool_call, extract_tool_call, parse_tool_call,
};
use anyhow::{Context, Result, anyhow, bail};
use std::env;
use std::io::{self, IsTerminal, Read, Write};
use std::time::{Duration, Instant};

const MAX_TOOL_STEPS: usize = 8;
const STATUS_CLEAR: &str = "\r\x1b[2K";

impl Cli {
    pub fn run(self) -> Result<()> {
        let stdin_is_terminal = io::stdin().is_terminal();
        let prompt = self.resolve_prompt(stdin_is_terminal)?;
        let config = Config::load().context("failed to load config")?;
        let request = self.build_request(&config, prompt)?;
        let host = self.resolve_host(&config)?;
        let model = request.model.clone();
        let effective_system_prompt =
            Self::effective_system_prompt(request.system.as_deref(), !self.no_tools);
        let client = OllamaClient::new(host.clone());
        let started_at = Instant::now();
        let final_response = if self.no_tools {
            self.run_without_tools(&client, &request, &effective_system_prompt)?
        } else {
            self.run_tool_loop(
                &client,
                &request,
                &effective_system_prompt,
                env::current_dir()?,
            )?
        };

        if self.verbose {
            self.print_verbose(
                &host,
                &model,
                &request,
                &effective_system_prompt,
                &final_response,
                started_at.elapsed(),
            )?;
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

    fn run_tool_loop(
        &self,
        client: &OllamaClient,
        request: &GenerateRequest,
        effective_system_prompt: &str,
        workspace_root: std::path::PathBuf,
    ) -> Result<GenerateResponse> {
        let mut prompt = Self::initial_tool_prompt(&request.prompt);

        for _ in 0..MAX_TOOL_STEPS {
            let step_request = GenerateRequest {
                model: request.model.clone(),
                prompt: prompt.clone(),
                system: Some(effective_system_prompt.to_owned()),
                temp: request.temp,
                stream: request.stream,
            };

            let turn = if request.stream {
                self.run_streaming_turn(client, &step_request)?
            } else {
                self.run_non_streaming_turn(client, &step_request)?
            };

            match turn {
                TurnOutcome::Final(response) => return Ok(response),
                TurnOutcome::Tool { call, preamble } => {
                    if Self::should_write_tool_status(request) {
                        let mut stdout = io::stdout().lock();
                        write!(stdout, "\r{}", Self::tool_status_message(&call))
                            .context("failed to write tool status")?;
                        stdout.flush().context("failed to flush tool status")?;
                    }

                    let tool_result = execute_tool_call(&workspace_root, &call)?;

                    if Self::should_write_tool_status(request) {
                        let mut stdout = io::stdout().lock();
                        write!(stdout, "{STATUS_CLEAR}").context("failed to clear tool status")?;
                        stdout.flush().context("failed to flush tool status")?;
                    }

                    if !preamble.is_empty() {
                        prompt.push_str("\n\nAssistant response before tool:\n");
                        prompt.push_str(&preamble);
                    }

                    prompt.push_str("\n\nAssistant tool request:\n");
                    prompt.push_str(&call.display());
                    prompt.push_str("\n\nTool result:\n");
                    prompt.push_str(&tool_result);
                    prompt.push_str(
                        "\n\nContinue. Use another TOOL line if needed, otherwise answer normally.",
                    );
                }
            }
        }

        bail!("tool loop exceeded maximum step count")
    }

    fn run_without_tools(
        &self,
        client: &OllamaClient,
        request: &GenerateRequest,
        effective_system_prompt: &str,
    ) -> Result<GenerateResponse> {
        let step_request = GenerateRequest {
            model: request.model.clone(),
            prompt: request.prompt.clone(),
            system: Some(effective_system_prompt.to_owned()),
            temp: request.temp,
            stream: request.stream,
        };

        if request.stream {
            self.run_streaming_turn_without_tools(client, &step_request)
        } else {
            self.run_non_streaming_turn_without_tools(client, &step_request)
        }
    }

    fn run_non_streaming_turn(
        &self,
        client: &OllamaClient,
        request: &GenerateRequest,
    ) -> Result<TurnOutcome> {
        let response = client.generate(request)?;
        if let Some(extracted) = extract_tool_call(&response.response)? {
            return Ok(TurnOutcome::Tool {
                call: extracted.call,
                preamble: extracted.preamble,
            });
        }

        self.write_final_response(&response)?;
        Ok(TurnOutcome::Final(response))
    }

    fn run_non_streaming_turn_without_tools(
        &self,
        client: &OllamaClient,
        request: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let response = client.generate(request)?;
        self.write_final_response(&response)?;
        Ok(response)
    }

    fn run_streaming_turn(
        &self,
        client: &OllamaClient,
        request: &GenerateRequest,
    ) -> Result<TurnOutcome> {
        let mut stdout = io::stdout().lock();
        let mut probe = StreamProbe::new();

        let response = client.generate_streaming(request, |chunk| {
            let action = probe.ingest(&chunk.response, chunk.done)?;
            if !action.flush.is_empty() {
                stdout
                    .write_all(action.flush.as_bytes())
                    .context("failed to write response to stdout")?;
                stdout.flush().context("failed to flush stdout")?;
            }
            Ok(if action.stop {
                StreamCallbackAction::Stop
            } else {
                StreamCallbackAction::Continue
            })
        })?;

        match probe.finish()? {
            StreamProbeResult::Tool { call, preamble } => Ok(TurnOutcome::Tool { call, preamble }),
            StreamProbeResult::PassThrough => {
                if !response.response.ends_with('\n') {
                    stdout
                        .write_all(b"\n")
                        .context("failed to write newline to stdout")?;
                    stdout.flush().context("failed to flush stdout")?;
                }
                Ok(TurnOutcome::Final(response))
            }
        }
    }

    fn run_streaming_turn_without_tools(
        &self,
        client: &OllamaClient,
        request: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let mut stdout = io::stdout().lock();
        let response = client.generate_streaming(request, |chunk| {
            if !chunk.response.is_empty() {
                stdout
                    .write_all(chunk.response.as_bytes())
                    .context("failed to write response to stdout")?;
                stdout.flush().context("failed to flush stdout")?;
            }
            Ok(StreamCallbackAction::Continue)
        })?;

        if !response.response.ends_with('\n') {
            stdout
                .write_all(b"\n")
                .context("failed to write newline to stdout")?;
            stdout.flush().context("failed to flush stdout")?;
        }

        Ok(response)
    }

    fn effective_system_prompt(existing: Option<&str>, tools_enabled: bool) -> String {
        match (existing, tools_enabled) {
            (Some(existing), true) => format!("{existing}\n\n{TOOL_SYSTEM_PROMPT}"),
            (Some(existing), false) => existing.to_owned(),
            (None, true) => TOOL_SYSTEM_PROMPT.to_owned(),
            (None, false) => String::new(),
        }
    }

    fn initial_tool_prompt(prompt: &str) -> String {
        format!("User request:\n{prompt}")
    }

    fn should_write_tool_status(request: &GenerateRequest) -> bool {
        request.stream
    }

    fn tool_status_message(call: &ToolCall) -> String {
        match call {
            ToolCall::Ls { path } => format!("* listing {path}..."),
            ToolCall::Read { path } => format!("* reading {path}..."),
            ToolCall::Stat { path } => format!("* stating {path}..."),
            ToolCall::Tree { path, .. } => format!("* walking {path}..."),
            ToolCall::Glob { pattern } => format!("* globbing {pattern}..."),
            ToolCall::Search { pattern, path } => format!("* searching {path} for {pattern}..."),
            ToolCall::Env => "* reading environment...".to_owned(),
        }
    }

    fn write_final_response(&self, response: &GenerateResponse) -> Result<()> {
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
        Ok(())
    }

    fn print_verbose(
        &self,
        host: &str,
        model: &str,
        request: &GenerateRequest,
        effective_system_prompt: &str,
        response: &GenerateResponse,
        elapsed: Duration,
    ) -> Result<()> {
        let mut stderr = io::stderr().lock();
        writeln!(stderr, "host: {host}").context("failed to write verbose output")?;
        writeln!(stderr, "model: {model}").context("failed to write verbose output")?;
        writeln!(stderr, "script_output: {}", !request.stream)
            .context("failed to write verbose output")?;
        if let Some(temp) = request.temp {
            writeln!(stderr, "temp: {temp}").context("failed to write verbose output")?;
        }
        writeln!(stderr, "system_prompt:\n{effective_system_prompt}")
            .context("failed to write verbose output")?;
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

enum TurnOutcome {
    Final(GenerateResponse),
    Tool { call: ToolCall, preamble: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StreamProbeResult {
    PassThrough,
    Tool { call: ToolCall, preamble: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StreamProbeAction {
    flush: String,
    stop: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StreamProbe {
    line_state: StreamLineState,
    line_buffer: String,
    preamble: String,
    tool_call: Option<ToolCall>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamLineState {
    Candidate,
    Passthrough,
}

impl StreamProbe {
    fn new() -> Self {
        Self {
            line_state: StreamLineState::Candidate,
            line_buffer: String::new(),
            preamble: String::new(),
            tool_call: None,
        }
    }

    fn ingest(&mut self, text: &str, done: bool) -> Result<StreamProbeAction> {
        let mut flush = String::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            match self.line_state {
                StreamLineState::Candidate => {
                    let split_at = remaining
                        .find('\n')
                        .map(|idx| idx + 1)
                        .unwrap_or(remaining.len());
                    self.line_buffer.push_str(&remaining[..split_at]);
                    remaining = &remaining[split_at..];

                    if self.line_buffer.ends_with('\n') {
                        let line = std::mem::take(&mut self.line_buffer);
                        match parse_tool_call(line.trim()) {
                            Ok(Some(call)) => {
                                self.tool_call = Some(call);
                                return Ok(StreamProbeAction { flush, stop: true });
                            }
                            Ok(None) | Err(_) => {
                                flush.push_str(&line);
                                self.preamble.push_str(&line);
                            }
                        }
                    } else if !self.current_line_could_be_tool() {
                        let line = std::mem::take(&mut self.line_buffer);
                        flush.push_str(&line);
                        self.preamble.push_str(&line);
                        self.line_state = StreamLineState::Passthrough;
                    }
                }
                StreamLineState::Passthrough => {
                    let split_at = remaining
                        .find('\n')
                        .map(|idx| idx + 1)
                        .unwrap_or(remaining.len());
                    let segment = &remaining[..split_at];
                    flush.push_str(segment);
                    self.preamble.push_str(segment);
                    remaining = &remaining[split_at..];

                    if segment.ends_with('\n') {
                        self.line_state = StreamLineState::Candidate;
                    }
                }
            }
        }

        if done && !self.line_buffer.is_empty() {
            let line = std::mem::take(&mut self.line_buffer);
            match parse_tool_call(line.trim()) {
                Ok(Some(call)) => {
                    self.tool_call = Some(call);
                    return Ok(StreamProbeAction { flush, stop: true });
                }
                Ok(None) | Err(_) => {
                    flush.push_str(&line);
                    self.preamble.push_str(&line);
                    self.line_state = StreamLineState::Candidate;
                }
            }
        }

        Ok(StreamProbeAction { flush, stop: false })
    }

    fn current_line_could_be_tool(&self) -> bool {
        let trimmed = self.line_buffer.trim_start();
        trimmed.is_empty() || "TOOL:".starts_with(trimmed) || trimmed.starts_with("TOOL:")
    }

    fn finish(self) -> Result<StreamProbeResult> {
        if let Some(call) = self.tool_call {
            Ok(StreamProbeResult::Tool {
                call,
                preamble: self.preamble,
            })
        } else {
            Ok(StreamProbeResult::PassThrough)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, StreamProbe, StreamProbeAction, StreamProbeResult};
    use crate::config::Config;
    use crate::tools::ToolCall;

    #[test]
    fn resolve_prompt_prefers_cli_prompt() {
        let cli = Cli {
            host: None,
            model: None,
            system: None,
            temp: None,
            no_tools: false,
            stream: true,
            verbose: false,
            prompt: Some("prompt".to_owned()),
        };

        assert_eq!(
            Cli::compose_prompt(cli.prompt.clone(), None).unwrap(),
            "prompt"
        );
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
            no_tools: false,
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
            no_tools: false,
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

    #[test]
    fn effective_system_prompt_appends_tool_instructions() {
        let prompt = Cli::effective_system_prompt(Some("be concise"), true);
        assert!(prompt.contains("be concise"));
        assert!(prompt.contains("TOOL: read"));
    }

    #[test]
    fn effective_system_prompt_skips_tool_instructions_when_disabled() {
        let prompt = Cli::effective_system_prompt(Some("be concise"), false);
        assert_eq!(prompt, "be concise");
    }

    #[test]
    fn effective_system_prompt_is_empty_without_system_or_tools() {
        let prompt = Cli::effective_system_prompt(None, false);
        assert!(prompt.is_empty());
    }

    #[test]
    fn initial_tool_prompt_wraps_user_request() {
        assert_eq!(
            Cli::initial_tool_prompt("summarize this"),
            "User request:\nsummarize this"
        );
    }

    #[test]
    fn tool_status_message_is_human_readable() {
        let msg = Cli::tool_status_message(&crate::tools::ToolCall::Read {
            path: "src/main.rs".to_owned(),
        });
        assert_eq!(msg, "* reading src/main.rs...");
    }

    #[test]
    fn tool_status_message_handles_env_tool() {
        let msg = Cli::tool_status_message(&crate::tools::ToolCall::Env);
        assert_eq!(msg, "* reading environment...");
    }

    #[test]
    fn writes_tool_status_only_when_streaming() {
        let streaming = crate::ollama::GenerateRequest {
            model: "llama3".to_owned(),
            prompt: "prompt".to_owned(),
            system: None,
            temp: None,
            stream: true,
        };
        let non_streaming = crate::ollama::GenerateRequest {
            stream: false,
            ..streaming.clone()
        };

        assert!(Cli::should_write_tool_status(&streaming));
        assert!(!Cli::should_write_tool_status(&non_streaming));
    }

    #[test]
    fn stream_probe_passes_through_normal_output() {
        let mut probe = StreamProbe::new();
        let action = probe.ingest("Hello", true).unwrap();

        assert_eq!(
            action,
            StreamProbeAction {
                flush: "Hello".to_owned(),
                stop: false
            }
        );

        assert_eq!(probe.finish().unwrap(), StreamProbeResult::PassThrough);
    }

    #[test]
    fn stream_probe_flushes_after_tool_prefix_mismatch() {
        let mut probe = StreamProbe::new();
        let action = probe.ingest("T", false).unwrap();
        assert_eq!(
            action,
            StreamProbeAction {
                flush: String::new(),
                stop: false
            }
        );

        let action = probe.ingest("his repository", false).unwrap();
        assert_eq!(
            action,
            StreamProbeAction {
                flush: "This repository".to_owned(),
                stop: false
            }
        );

        assert_eq!(probe.finish().unwrap(), StreamProbeResult::PassThrough);
    }

    #[test]
    fn stream_probe_detects_split_tool_prefix() {
        let mut probe = StreamProbe::new();
        let action = probe.ingest("TOO", false).unwrap();
        assert_eq!(
            action,
            StreamProbeAction {
                flush: String::new(),
                stop: false
            }
        );
        let action = probe.ingest("L: read|src/main.rs", true).unwrap();
        assert_eq!(
            action,
            StreamProbeAction {
                flush: String::new(),
                stop: true
            }
        );
        assert_eq!(
            probe.finish().unwrap(),
            StreamProbeResult::Tool {
                call: ToolCall::Read {
                    path: "src/main.rs".to_owned()
                },
                preamble: String::new()
            }
        );
    }

    #[test]
    fn stream_probe_uses_later_standalone_tool_line() {
        let mut probe = StreamProbe::new();
        let action = probe
            .ingest("I should inspect the README first.\n", false)
            .unwrap();
        assert_eq!(
            action,
            StreamProbeAction {
                flush: "I should inspect the README first.\n".to_owned(),
                stop: false
            }
        );
        let action = probe.ingest("TOOL: read|README.md", true).unwrap();
        assert_eq!(
            action,
            StreamProbeAction {
                flush: String::new(),
                stop: true
            }
        );

        assert_eq!(
            probe.finish().unwrap(),
            StreamProbeResult::Tool {
                call: ToolCall::Read {
                    path: "README.md".to_owned()
                },
                preamble: "I should inspect the README first.\n".to_owned()
            }
        );
    }

    #[test]
    fn stream_probe_ignores_inline_tool_text() {
        let mut probe = StreamProbe::new();
        let action = probe
            .ingest("I could say TOOL: read|README.md later.", true)
            .unwrap();

        assert_eq!(
            action,
            StreamProbeAction {
                flush: "I could say TOOL: read|README.md later.".to_owned(),
                stop: false
            }
        );

        assert_eq!(probe.finish().unwrap(), StreamProbeResult::PassThrough);
    }

    #[test]
    fn stream_probe_ignores_invalid_tool_line() {
        let mut probe = StreamProbe::new();
        let action = probe.ingest("TOOL: nope|README.md", true).unwrap();

        assert_eq!(
            action,
            StreamProbeAction {
                flush: "TOOL: nope|README.md".to_owned(),
                stop: false
            }
        );
        assert_eq!(probe.finish().unwrap(), StreamProbeResult::PassThrough);
    }

    #[test]
    fn stream_probe_discards_trailing_bytes_after_tool_line() {
        let mut probe = StreamProbe::new();
        let action = probe
            .ingest("Thinking...\nTOOL: read|README.md\nignored", false)
            .unwrap();

        assert_eq!(
            action,
            StreamProbeAction {
                flush: "Thinking...\n".to_owned(),
                stop: true
            }
        );
        assert_eq!(
            probe.finish().unwrap(),
            StreamProbeResult::Tool {
                call: ToolCall::Read {
                    path: "README.md".to_owned()
                },
                preamble: "Thinking...\n".to_owned()
            }
        );
    }
}
