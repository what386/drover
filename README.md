# drover

`drover` is a small CLI for sending prompts to a local Ollama instance.

It supports:
- prompt input as a positional argument
- prompt input from `stdin`
- combined prompt plus `stdin` context
- model, system prompt, and temperature overrides
- host override for one-off Ollama targets
- interactive or script-friendly buffered output
- read-only filesystem tool invocation during a run
- a simple TOML config file

## Build

```bash
cargo build
```

Run it with:

```bash
cargo run -- "Write a haiku about Rust"
```

## Usage

```bash
drover "prompt"
cat file.txt | drover
cat file.txt | drover "what is this about?"

drover --model llama3 "prompt"
drover --host http://localhost:11434 "prompt"
drover --system "you are a poet" "prompt"
drover --temp 0.7 "prompt"
drover --no-tools "prompt"
drover --script-output "prompt"
drover --verbose "prompt"

drover --help
drover --version
```

## Config

`drover` reads its config from:

```text
~/.config/drover/config.toml
```

If the file does not exist, `drover` creates it with defaults.

Example:

```toml
model = "llama3"
host = "http://localhost:11434"
temp = 0.7
```

CLI flags override config values for the current invocation.

## Notes

- `model` should match the exact Ollama model name you see in `ollama list`
- `--host` overrides the configured Ollama server for a single run
- if both stdin and a prompt are provided, the prompt is treated as the instruction and stdin is appended as `Input:` context
- the model can request read-only tools using `TOOL: read|<path>`, `TOOL: ls|<path>`, `TOOL: stat|<path>`, `TOOL: tree|<path>|<depth>`, `TOOL: glob|<pattern>`, `TOOL: search|<pattern>|<path>`, and `TOOL: env`
- use `glob` to find files by name or extension, `search` to match file contents, `stat` to inspect metadata, and `env` to inspect execution context
- `--no-tools` disables tool prompting and ignores unprompted `TOOL:` output as plain model text
- `--system` is a per-command override
- `--verbose` prints request and timing details to `stderr`
- `--script-output` waits for the full response before printing and suppresses transient callbacks such as tool status updates

## Requirements

- Rust and Cargo
- a running Ollama instance
- at least one local Ollama model installed
