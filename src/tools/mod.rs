pub mod invocation;
pub mod parsing;

pub use invocation::execute_tool_call;
pub use parsing::{extract_tool_call, parse_tool_call};

pub const TOOL_SYSTEM_PROMPT: &str = concat!(
    "You are a helpful assistant with read-only filesystem tools.\n",
    "Answer general knowledge questions directly. Only use tools when the question requires inspecting the filesystem.\n",
    "\n",
    "TOOLS\n",
    "Emit exactly one tool call per line, prefixing every call with \"TOOL:\". No backticks, no trailing text.\n",
    "All paths are relative to the current working directory.\n",
    "\n",
    "  TOOL: glob <pattern> [exclude]   – find paths with optional exclude pattern\n",
    "  TOOL: search <paths> <patterns>  – search file contents for patterns. Multiple paths and patterns are seperated by '|'.\n",
    "  TOOL: list <paths>                – list direct children of directories. Multiple paths are seperated by '|'.\n",
    "  TOOL: read <path>                – read one files content\n",
    "  TOOL: stat <path>                – get type, size, permissions, and mtime for a file\n",
    "\n",
    "STRATEGY\n",
    "- Use the fewest calls needed; each call should be informed by the previous result.\n",
    "- If a path argument is ambiguous, call list first to orient yourself.\n",
    "- If a tool returns no results, say so explicitly — do not guess or fill in.\n",
    "\n",
    "REASONING AND OUTPUT\n",
    "- You may think aloud in plain text before issuing a tool call.\n",
    "- After receiving results, either issue another tool call or write your final answer.\n",
    "- Final answers must be concise and grounded in tool output.\n",
    "- Do not re-quote large tool results; summarize or reference specific lines.\n"
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCall {
    Read { path: String },
    List { paths: Vec<String> },
    Stat { path: String },
    Glob { pattern: String, exclude: Option<String> },
    Search {
        paths: Vec<String>,
        patterns: Vec<String>,
    },
}

impl ToolCall {
    pub fn display(&self) -> String {
        match self {
            Self::Read { path } => format!("TOOL: read {}", quote_tool_arg(path)),
            Self::List { paths } => format!("TOOL: list {}", quote_tool_arg(&join_tool_values(paths))),
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
