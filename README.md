# drover

`drover` is a CLI tool for interacting with a local Ollama instance. It allows you to send prompts, leveraging various configurations and even tools for filesystem interaction.

## Features

- **Scriptable Input:** Supports prompts via positional arguments, standard input (`stdin`), or a combination of both.
- **Filesystem Tools:** Enables the model to interact with the filesystem (optional), such as reading files, listing directories and searching for text, allowing it to process local files as context.

## Installation

**Prerequisites:**

- **Ollama:** You must have a running Ollama instance ([https://ollama.com/](https://ollama.com/)).
- **Ollama Model:** At least one Ollama model must be downloaded (e.g., `ollama pull llama3`).

You can install the program via cargo:

```bash
cargo install drover
```

**Building from Source:**

```bash
cargo build --release
```

After building, the compiled binary is located in `./target/debug/drover`.

## Usage

```bash
# Basic prompt
drover "Tell me a story."

# Prompt with stdin context
cat my_document.txt | drover "Summarize this document."

# Example with overrides
drover --profile local --system "You speak like Shakespeare." --temp 2 --tools false "Explain the concept of recursion."
```

## Configuration

`drover` loads its configuration from `~/.config/drover/config.toml`. If the file doesn't exist, it will use the default values.

**Default Configuration:**

```toml
[default]
model = "llama3"
host = "http://localhost:11434"
temp = 0.7
allow_tools = false
```

**CLI flags will override values defined in the config file.**

`drover` also accepts `--profile` / `-p` to select a config profile name. If omitted, it defaults to `default`. The profile name is parsed and carried through the CLI now so it can be wired into profile-based config selection later.

## Tool Usage

When the `--tools` flag is enabled (or `allow_tools = true` in the config file), `drover` lets the model interact with your filesystem.

**Supported Tools:**

| Tool     | Description                     |
| -------- | ------------------------------- |
| `read`   | Reads the contents of a file.   |
| `list`   | Lists directory contents.       |
| `stat`   | Returns file metadata.          |
| `glob`   | Finds files matching a pattern. |
| `search` | Searches for text within files. |

**Example:**

```bash
drover --tools true "Summarize key features from this repository"
```

This would instruct `drover` to allow the LLM to request the tool to read files and list directories to answer the question.
