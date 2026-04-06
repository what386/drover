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

If the file does not exist, `drover` creates it with defaults:

```toml
model = "llama3"
host = "http://localhost:11434"
temp = 0.7
```

Example:

```toml
model = "llama3"
host = "http://localhost:11434"
temp = 0.7
```

CLI flags override config values for the current invocation.

## Using Tools

Drover allows the model to interact with the filesystem, searching, listing, and reading files. Useful tool outputs help improve generation quality. You can add command invocations within your prompt, like so: `TOOL: read file.txt`. When provided, `drover` sends this invocation and presents the contents in subsequent prompt iterations until you ask the model to avoid tool invocations through `--no-tools`.

For example:

```bash
cat file.txt | drover "Summarize key points from this file"
```

This sends `file.txt` to Ollama and uses the contents in answering the prompt. It supports the following tools:

- `TOOL: read <path>`: Reads the contents of a file.
- `TOOL: ls <path>`: Lists the contents of a directory.
- `TOOL: stat <path>`: Returns metadata about a file (size, modified time, etc.).
- `TOOL: tree <path> [depth]`: Recursively lists a directory structure up to a specified depth. When omitted, `depth` defaults to `2`.
- `TOOL: glob <pattern> [exclude]`: Finds files matching a pattern with an optional exclude glob (e.g., `TOOL: glob **/*.rs target/**/*`).
- `TOOL: search <pattern> <path>`: Searches for text within files.

If an argument contains spaces, quote it, for example `TOOL: read "my dir/file.txt"` or `TOOL: search "my handler fn" src`.
