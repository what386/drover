# drover

`drover` is a small CLI for sending prompts to a local Ollama instance.

It supports:

- Prompt input as a positional argument
- Prompt input from `stdin`
- Combined prompt plus `stdin` context
- Model, system prompt, and temperature overrides
- Host override for one-off Ollama targets
- Interactive or script-friendly buffered output
- Read-only filesystem tool invocation during a run
- A simple TOML config file

## Installation

1.  **Requirements:**
    - A running Ollama instance ([https://ollama.com/](https://ollama.com/))
    - At least one local Ollama model installed (via `ollama pull <model_name>`).

2.  **Build:**

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
drover --tools false "prompt"
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

The config file is not created automatically. Create it yourself if you want persistent defaults.

Default values:

```toml
model = "llama3"
host = "http://localhost:11434"
temp = 0.7
allow_tools = false
```

Example:

```toml
model = "llama3"
host = "http://localhost:11434"
temp = 0.7
allow_tools = false
```

CLI flags override config values for the current invocation, including `--tools true|false`.

## Using Tools

When tools are enabled, drover allows the model to interact with the filesystem during a run. Tool access is controlled by `allow_tools` in config and can be overridden per invocation with `--tools true` or `--tools false`.

Tool calls use this format:

```text
TOOL: <name> <arg1> <arg2> ...
```

Rules:

- Emit exactly one tool call on a single line.
- Arguments are space-separated.
- Multiple values inside one argument use `|`.
- All paths are relative to the current working directory.

For example:

```bash
cat file.txt | drover --tools true "Summarize key points from this file"
```

This sends `file.txt` to Ollama and uses the contents in answering the prompt. It supports the following tools:

- `TOOL: read <path>`: Reads the contents of a file.
- `TOOL: list <paths>`: Lists directory contents. Multiple paths go in one argument separated by `|`.
- `TOOL: stat <path>`: Returns metadata about a file (size, modified time, etc.).
- `TOOL: glob <pattern> [exclude]`: Finds files matching a pattern with an optional exclude glob (e.g., `TOOL: glob **/*.rs target/**/*`).
- `TOOL: search <paths> <patterns>`: Searches for text within files. Multiple paths or patterns go in one argument separated by `|`.

If an argument contains spaces, quote it. Examples:

- `TOOL: read "my dir/file.txt"`
- `TOOL: list "src|tests/my dir"`
- `TOOL: search "src|tests" "my handler fn|other pattern"`
