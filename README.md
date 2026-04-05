# drover

`drover` is a small CLI for sending prompts to a local Ollama instance.

It supports:
- prompt input as a positional argument
- prompt input from `stdin`
- model, system prompt, and temperature overrides
- host override for one-off Ollama targets
- streaming or non-streaming output
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

drover --model llama3 "prompt"
drover --host http://localhost:11434 "prompt"
drover --system "you are a poet" "prompt"
drover --temp 0.7 "prompt"
drover --no-stream "prompt"
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
- `--system` is a per-command override
- `--verbose` prints request and timing details to `stderr`
- `--no-stream` waits for the full response before printing

## Requirements

- Rust and Cargo
- a running Ollama instance
- at least one local Ollama model installed
