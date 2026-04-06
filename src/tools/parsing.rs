use crate::tools::{ExtractedToolCall, ToolCall};
use anyhow::{Result, anyhow, bail};

fn expect_args<'a>(
    parts: &'a [String],
    tool: &str,
    required_args: &[&str],
    max_args: usize,
) -> Result<&'a [String]> {
    let args = &parts[1..];

    if args.len() < required_args.len() {
        let missing = required_args[args.len()];
        bail!("missing {missing} for `{tool}` tool");
    }

    if args.len() > max_args {
        bail!("too many arguments for `{tool}` tool");
    }

    Ok(args)
}

fn tokenize_tool_args(input: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => in_quotes = !in_quotes,
            '\\' if in_quotes => {
                let escaped = chars
                    .next()
                    .ok_or_else(|| anyhow!("unterminated escape sequence in quoted argument"))?;
                match escaped {
                    '"' | '\\' => current.push(escaped),
                    'n' => current.push('\n'),
                    't' => current.push('\t'),
                    other => {
                        current.push('\\');
                        current.push(other);
                    }
                }
            }
            c if c.is_whitespace() && !in_quotes => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                while chars.next_if(|next| next.is_whitespace()).is_some() {}
            }
            _ => current.push(ch),
        }
    }

    if in_quotes {
        bail!("unterminated quoted argument");
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

pub fn parse_tool_call(output: &str) -> Result<Option<ToolCall>> {
    let trimmed = output.trim();
    if trimmed.is_empty() || !trimmed.starts_with("TOOL:") {
        return Ok(None);
    }

    let body = trimmed["TOOL:".len()..].trim();
    let parts = tokenize_tool_args(body)?;

    let command = parts
        .first()
        .map(String::as_str)
        .ok_or_else(|| anyhow!("missing tool name after `TOOL:`"))?;
    if command.contains('|') {
        bail!("legacy `|`-delimited tool syntax is not supported");
    }

    let call = match command {
        "read" => {
            let args = expect_args(&parts, "read", &["path"], 1)?;
            ToolCall::Read {
                path: args[0].clone(),
            }
        }
        "list" => {
            let args = expect_args(&parts, "list", &["paths"], 1)?;
            ToolCall::List {
                paths: split_multi_value_arg(&args[0], "list", "paths")?,
            }
        }
        "stat" => {
            let args = expect_args(&parts, "stat", &["path"], 1)?;
            ToolCall::Stat {
                path: args[0].clone(),
            }
        }
        "glob" => {
            let args = expect_args(&parts, "glob", &["pattern"], 2)?;
            ToolCall::Glob {
                pattern: args[0].clone(),
                exclude: args.get(1).map(String::to_owned),
            }
        }
        "search" => {
            let args = expect_args(&parts, "search", &["paths", "patterns"], 2)?;
            ToolCall::Search {
                paths: split_multi_value_arg(&args[0], "search", "paths")?,
                patterns: split_multi_value_arg(&args[1], "search", "patterns")?,
            }
        }
        _ => bail!("unknown tool: {command}"),
    };

    Ok(Some(call))
}

fn split_multi_value_arg(input: &str, tool: &str, arg_name: &str) -> Result<Vec<String>> {
    let values = input
        .split('|')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    if values.is_empty() {
        bail!("missing {arg_name} for `{tool}` tool");
    }

    Ok(values)
}

pub fn extract_tool_call(output: &str) -> Result<Option<ExtractedToolCall>> {
    let mut preamble = String::new();

    for segment in output.split_inclusive('\n') {
        let trimmed = segment.trim();
        match parse_tool_call(trimmed) {
            Ok(Some(call)) => return Ok(Some(ExtractedToolCall { preamble, call })),
            Ok(None) | Err(_) => preamble.push_str(segment),
        }
    }

    if output.is_empty() || output.ends_with('\n') {
        return Ok(None);
    }

    let trailing = output
        .rsplit_once('\n')
        .map(|(_, tail)| tail)
        .unwrap_or(output);
    if preamble.len() == output.len() {
        return Ok(None);
    }

    match parse_tool_call(trailing.trim()) {
        Ok(Some(call)) => {
            let keep_len = output.len() - trailing.len();
            Ok(Some(ExtractedToolCall {
                preamble: output[..keep_len].to_owned(),
                call,
            }))
        }
        Ok(None) | Err(_) => Ok(None),
    }
}



#[cfg(test)]
mod tests {
    use super::{extract_tool_call, parse_tool_call};
    use crate::tools::{ExtractedToolCall, ToolCall};

    #[test]
    fn parses_quoted_ls_path() {
        let call = parse_tool_call("TOOL: list \"my dir\"").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::List {
                paths: vec!["my dir".to_owned()]
            }
        );
    }

    #[test]
    fn parses_glob_tool_call_with_exclude() {
        let call = parse_tool_call("TOOL: glob **/*.rs target/**/*").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Glob {
                pattern: "**/*.rs".to_owned(),
                exclude: Some("target/**/*".to_owned()),
            }
        );
    }

    #[test]
    fn parses_search_with_spaces_in_pattern() {
        let call = parse_tool_call("TOOL: search src \"my handler fn\"")
            .unwrap()
            .unwrap();
        assert_eq!(
            call,
            ToolCall::Search {
                paths: vec!["src".to_owned()],
                patterns: vec!["my handler fn".to_owned()]
            }
        );
    }

    #[test]
    fn parses_multi_path_list_call() {
        let call = parse_tool_call("TOOL: list \"src|tests/my dir\"").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::List {
                paths: vec!["src".to_owned(), "tests/my dir".to_owned()]
            }
        );
    }

    #[test]
    fn parses_multi_path_multi_pattern_search() {
        let call = parse_tool_call("TOOL: search \"src|tests\" \"needle one|needle two\"")
            .unwrap()
            .unwrap();
        assert_eq!(
            call,
            ToolCall::Search {
                paths: vec!["src".to_owned(), "tests".to_owned()],
                patterns: vec!["needle one".to_owned(), "needle two".to_owned()]
            }
        );
    }

    #[test]
    fn parses_path_with_spaces() {
        let call = parse_tool_call("TOOL: read \"my dir/my file.rs\"")
            .unwrap()
            .unwrap();
        assert_eq!(
            call,
            ToolCall::Read {
                path: "my dir/my file.rs".to_owned()
            }
        );
    }

    #[test]
    fn parses_quoted_arg_with_escaped_quote() {
        let call = parse_tool_call(r#"TOOL: read "my \"quoted\" file.rs""#)
            .unwrap()
            .unwrap();
        assert_eq!(
            call,
            ToolCall::Read {
                path: "my \"quoted\" file.rs".to_owned()
            }
        );
    }

    #[test]
    fn ignores_non_tool_output() {
        assert!(parse_tool_call("normal answer").unwrap().is_none());
    }

    #[test]
    fn extracts_first_tool_call_and_preamble_from_multiline_response() {
        let extracted = extract_tool_call("Thinking...\nTOOL: read README.md\nTOOL: list src")
            .unwrap()
            .unwrap();
        assert_eq!(
            extracted,
            ExtractedToolCall {
                preamble: "Thinking...\n".to_owned(),
                call: ToolCall::Read {
                    path: "README.md".to_owned()
                }
            }
        );
    }

    #[test]
    fn ignores_inline_tool_text_when_extracting_tool_call() {
        let call = extract_tool_call("I might use TOOL: read README.md later.").unwrap();
        assert!(call.is_none());
    }

    #[test]
    fn ignores_invalid_tool_lines_when_extracting_tool_call() {
        let call = extract_tool_call("TOOL: nope README.md\nfinal answer").unwrap();
        assert!(call.is_none());
    }

    #[test]
    fn extracts_tool_call_after_invalid_tool_like_line() {
        let extracted = extract_tool_call("TOOL: nope README.md\nThinking...\nTOOL: list src")
            .unwrap()
            .unwrap();
        assert_eq!(
            extracted,
            ExtractedToolCall {
                preamble: "TOOL: nope README.md\nThinking...\n".to_owned(),
                call: ToolCall::List { paths: vec!["src".to_owned()] }
            }
        );
    }

    #[test]
    fn ignores_empty_input() {
        assert!(parse_tool_call("").unwrap().is_none());
    }

    #[test]
    fn rejects_missing_path_for_read() {
        let err = parse_tool_call("TOOL: read").unwrap_err();
        assert_eq!(err.to_string(), "missing path for `read` tool");
    }

    #[test]
    fn rejects_legacy_pipe_syntax() {
        let err = parse_tool_call("TOOL: read|note.txt").unwrap_err();
        assert_eq!(err.to_string(), "legacy `|`-delimited tool syntax is not supported");
    }

    #[test]
    fn rejects_unknown_tool() {
        let err = parse_tool_call("TOOL: write foo.txt").unwrap_err();
        assert!(err.to_string().contains("unknown tool"));
    }

    #[test]
    fn rejects_too_many_args_for_ls() {
        let err = parse_tool_call("TOOL: list src extra").unwrap_err();
        assert!(err.to_string().contains("too many arguments"));
    }

    #[test]
    fn rejects_unterminated_quoted_argument() {
        let err = parse_tool_call("TOOL: read \"unterminated").unwrap_err();
        assert_eq!(err.to_string(), "unterminated quoted argument");
    }

    #[test]
    fn parses_quoted_glob_exclude() {
        let call = parse_tool_call("TOOL: glob **/*.rs \"target generated/**/*\"")
            .unwrap()
            .unwrap();
        assert_eq!(
            call,
            ToolCall::Glob {
                pattern: "**/*.rs".to_owned(),
                exclude: Some("target generated/**/*".to_owned()),
            }
        );
    }
}
