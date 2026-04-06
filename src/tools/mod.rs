pub mod invocation;
pub mod parsing;

pub use invocation::execute_tool_call;
pub use parsing::{extract_tool_call, parse_tool_call};

pub const TOOL_SYSTEM_PROMPT: &str = concat!(
    "You have access to read-only filesystem tools in your current environment.\n",
    "Use them when needed to inspect files or directories.\n",
    "Answer general knowledge questions directly without using tools.\n",
    "\n",
    "TOOLS\n",
    "Emit tool calls as a single line:\n",
    "TOOL: <name> <arg1> <arg2> ...\n",
    "- No backticks or extra text on the line.\n",
    "- Arguments are space-separated.\n",
    "- Multiple values within one argument use '|'.\n",
    "- All paths are relative to the current working directory.\n",
    "- Absolute paths (e.g. /home/$(USER)/Documents) are forbidden.\n",
    "\n",
    "Available tools:\n",
    "  glob <pattern> [exclude]   – find matching paths\n",
    "  search <paths> <patterns>  – search file contents\n",
    "  list <paths>               – list directory contents\n",
    "  read <path>                – read file contents\n",
    "  stat <path>                – file metadata (type, size, permissions, mtime)\n",
    "\n",
    "STRATEGY\n",
    "- Use the minimum number of calls.\n",
    "- Each call should build on previous results.\n",
    "- If unsure of a path, use `list` first.\n",
    "- If a tool returns nothing, state that explicitly.\n",
    "\n",
    "OUTPUT\n",
    "- You may include brief reasoning before a tool call.\n",
    "- After a tool result: either call another tool or answer.\n",
    "- Final answers must be concise and based only on tool results.\n",
    "- Do not repeat large outputs; summarize relevant information.\n"
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCall {
    Read {
        path: String,
    },
    List {
        paths: Vec<String>,
    },
    Stat {
        path: String,
    },
    Glob {
        pattern: String,
        exclude: Option<String>,
    },
    Search {
        paths: Vec<String>,
        patterns: Vec<String>,
    },
}

impl ToolCall {
    pub fn display(&self) -> String {
        match self {
            Self::Read { path } => format!("TOOL: read {}", quote_tool_arg(path)),
            Self::List { paths } => {
                format!("TOOL: list {}", quote_tool_arg(&join_tool_values(paths)))
            }
            Self::Stat { path } => format!("TOOL: stat {}", quote_tool_arg(path)),
            Self::Glob { pattern, exclude } => match exclude {
                Some(exclude) => {
                    format!(
                        "TOOL: glob {} {}",
                        quote_tool_arg(pattern),
                        quote_tool_arg(exclude)
                    )
                }
                None => format!("TOOL: glob {}", quote_tool_arg(pattern)),
            },
            Self::Search { paths, patterns } => {
                format!(
                    "TOOL: search {} {}",
                    quote_tool_arg(&join_tool_values(paths)),
                    quote_tool_arg(&join_tool_values(patterns))
                )
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedToolCall {
    pub preamble: String,
    pub call: ToolCall,
}

fn quote_tool_arg(arg: &str) -> String {
    if arg.is_empty()
        || arg
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '"' | '\\'))
    {
        let escaped = arg.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{escaped}\"")
    } else {
        arg.to_owned()
    }
}

fn join_tool_values(values: &[String]) -> String {
    values.join("|")
}

#[cfg(test)]
mod tests {
    use super::ToolCall;

    #[test]
    fn display_quotes_arguments_with_spaces() {
        let call = ToolCall::Search {
            paths: vec!["src/my dir".to_owned(), "tests".to_owned()],
            patterns: vec!["my handler fn".to_owned(), "other".to_owned()],
        };
        assert_eq!(
            call.display(),
            "TOOL: search \"src/my dir|tests\" \"my handler fn|other\""
        );
    }

    #[test]
    fn display_escapes_embedded_quotes() {
        let call = ToolCall::Read {
            path: "my \"quoted\" file.rs".to_owned(),
        };
        assert_eq!(call.display(), r#"TOOL: read "my \"quoted\" file.rs""#);
    }
}
