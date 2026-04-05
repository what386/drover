use anyhow::{Context, Result, anyhow, bail};
use globset::{GlobBuilder, GlobMatcher};
use std::fs;
use std::path::{Path, PathBuf};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

const MAX_TOOL_OUTPUT_BYTES: usize = 16 * 1024;
const MAX_GLOB_MATCHES: usize = 500;
const MAX_SEARCH_MATCHES: usize = 200;
const MAX_SEARCH_DEPTH: usize = 8;

pub const TOOL_SYSTEM_PROMPT: &str = concat!(
    "You are a local assistant with read-only filesystem tools.\n",
    "Never guess file or directory contents — always call a tool first.\n",
    "You may think briefly in plain text before using a tool.\n",
    "When you call a tool, put the actual invocation on its own standalone line:\n",
    "TOOL: ls|<path>            # list children of one directory\n",
    "TOOL: tree|<path>|<depth>  # recursive directory listing up to <depth> levels\n",
    "TOOL: glob|<pattern>       # find paths matching a glob (e.g. **/*.rs)\n",
    "TOOL: read|<path>          # get the contents of one file\n",
    "TOOL: search|<pattern>|<path> # grep-like search within files under <path>\n",
    "TOOL: stat|<path>          # get file metadata: size, type, permissions, etc\n",
    "TOOL: env                  # show current CWD + environment variables\n",
    "Anything after a tool line may be ignored.\n",
    "Paths must be relative. After receiving results, call another tool or answer normally.\n",
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCall {
    Read { path: String },
    Ls { path: String },
    Stat { path: String },
    Tree { path: String, depth: usize },
    Glob { pattern: String },
    Search { pattern: String, path: String },
    Env,
}

impl ToolCall {
    pub fn display(&self) -> String {
        match self {
            Self::Read { path } => format!("TOOL: read|{path}"),
            Self::Ls { path } => format!("TOOL: ls|{path}"),
            Self::Stat { path } => format!("TOOL: stat|{path}"),
            Self::Tree { path, depth } => format!("TOOL: tree|{path}|{depth}"),
            Self::Glob { pattern } => format!("TOOL: glob|{pattern}"),
            Self::Search { pattern, path } => format!("TOOL: search|{pattern}|{path}"),
            Self::Env => "TOOL: env".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedToolCall {
    pub preamble: String,
    pub call: ToolCall,
}

pub fn parse_tool_call(output: &str) -> Result<Option<ToolCall>> {
    let trimmed = output.trim();
    if trimmed.is_empty() || !trimmed.starts_with("TOOL:") {
        return Ok(None);
    }

    let body = trimmed["TOOL:".len()..].trim();
    let mut parts: Vec<&str> = body.split('|').map(str::trim).collect();
    while matches!(parts.last(), Some(part) if part.is_empty()) {
        parts.pop();
    }

    let command = parts
        .first()
        .copied()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("missing tool name after `TOOL:`"))?;

    let call = match command {
        "read" => {
            let path = parts
                .get(1)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("missing path for `read` tool"))?;
            if parts.len() > 2 {
                bail!("too many arguments for `read` tool");
            }
            ToolCall::Read {
                path: path.to_owned(),
            }
        }
        "ls" => {
            let path = parts
                .get(1)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("missing path for `ls` tool"))?;
            if parts.len() > 2 {
                bail!("too many arguments for `ls` tool");
            }
            ToolCall::Ls {
                path: path.to_owned(),
            }
        }
        "stat" => {
            let path = parts
                .get(1)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("missing path for `stat` tool"))?;
            if parts.len() > 2 {
                bail!("too many arguments for `stat` tool");
            }
            ToolCall::Stat {
                path: path.to_owned(),
            }
        }
        "tree" => {
            let path = parts
                .get(1)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("missing path for `tree` tool"))?;
            let depth = parts
                .get(2)
                .copied()
                .filter(|s| !s.is_empty())
                .unwrap_or("3")
                .parse::<usize>()
                .map_err(|_| anyhow!("depth must be a positive integer"))?;
            let depth = depth.min(MAX_SEARCH_DEPTH);
            if parts.len() > 3 {
                bail!("too many arguments for `tree` tool");
            }
            ToolCall::Tree {
                path: path.to_owned(),
                depth,
            }
        }
        "glob" => {
            let pattern = parts
                .get(1)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("missing pattern for `glob` tool"))?;
            if parts.len() > 2 {
                bail!("too many arguments for `glob` tool");
            }
            ToolCall::Glob {
                pattern: pattern.to_owned(),
            }
        }
        "search" => {
            let pattern = parts
                .get(1)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("missing pattern for `search` tool"))?;
            let path = parts
                .get(2)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("missing path for `search` tool"))?;
            if parts.len() > 3 {
                bail!("too many arguments for `search` tool");
            }
            ToolCall::Search {
                pattern: pattern.to_owned(),
                path: path.to_owned(),
            }
        }
        "env" => {
            if parts.len() > 1 {
                bail!("too many arguments for `env` tool");
            }
            ToolCall::Env
        }
        _ => bail!("unknown tool: {command}"),
    };

    Ok(Some(call))
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

pub fn execute_tool_call(base_dir: &Path, call: &ToolCall) -> Result<String> {
    match call {
        ToolCall::Read { path } => read_file(base_dir, path),
        ToolCall::Ls { path } => list_path(base_dir, path),
        ToolCall::Stat { path } => stat_path(base_dir, path),
        ToolCall::Tree { path, depth } => tree_path(base_dir, path, *depth),
        ToolCall::Glob { pattern } => glob_paths(base_dir, pattern),
        ToolCall::Search { pattern, path } => search_path(base_dir, pattern, path),
        ToolCall::Env => environment_info(base_dir),
    }
}

fn read_file(base_dir: &Path, path: &str) -> Result<String> {
    let resolved = resolve_path(base_dir, path)?;
    let content = fs::read_to_string(&resolved)
        .with_context(|| format!("failed to read file: {}", resolved.display()))?;
    Ok(truncate_output(
        format!(
            "READ {}\n{}",
            display_relative(base_dir, &resolved),
            content
        ),
        path,
    ))
}

fn list_path(base_dir: &Path, path: &str) -> Result<String> {
    let resolved = resolve_path(base_dir, path)?;
    let metadata = fs::metadata(&resolved)
        .with_context(|| format!("failed to inspect path: {}", resolved.display()))?;

    if metadata.is_file() {
        return Ok(format!(
            "LS {}\nFILE",
            display_relative(base_dir, &resolved)
        ));
    }

    let mut entries = fs::read_dir(&resolved)
        .with_context(|| format!("failed to list directory: {}", resolved.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to list directory: {}", resolved.display()))?;
    entries.sort_by_key(|e| e.file_name());

    let mut lines = vec![format!("LS {}", display_relative(base_dir, &resolved))];
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = if entry.path().is_dir() { "DIR" } else { "FILE" };
        lines.push(format!("{kind} {name}"));
    }

    Ok(truncate_output(lines.join("\n"), path))
}

fn stat_path(base_dir: &Path, path: &str) -> Result<String> {
    let resolved = resolve_path(base_dir, path)?;
    let metadata = fs::metadata(&resolved)
        .with_context(|| format!("failed to inspect path: {}", resolved.display()))?;
    let relative = display_relative(base_dir, &resolved);
    let path_type = if metadata.is_dir() { "DIR" } else { "FILE" };
    let modified = metadata
        .modified()
        .ok()
        .and_then(format_system_time)
        .unwrap_or_else(|| "unknown".to_owned());

    let lines = [
        format!("STAT {relative}"),
        format!("path: {relative}"),
        format!("type: {path_type}"),
        format!("size_bytes: {}", metadata.len()),
        format!("modified_utc: {modified}"),
        format!("permissions: {}", format_permissions(&metadata)),
    ];

    Ok(truncate_output(lines.join("\n"), path))
}

fn tree_path(base_dir: &Path, path: &str, max_depth: usize) -> Result<String> {
    let resolved = resolve_path(base_dir, path)?;
    let rel = display_relative(base_dir, &resolved);
    let mut lines = vec![format!("TREE {rel} (depth {max_depth})")];
    collect_tree(&resolved, 0, max_depth, &mut lines)?;
    Ok(truncate_output(lines.join("\n"), path))
}

fn collect_tree(
    path: &Path,
    depth: usize,
    max_depth: usize,
    lines: &mut Vec<String>,
) -> Result<()> {
    if depth >= max_depth {
        return Ok(());
    }

    let mut entries = fs::read_dir(path)
        .with_context(|| format!("failed to list directory: {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to list directory: {}", path.display()))?;
    entries.sort_by_key(|e| e.file_name());

    let indent = "  ".repeat(depth + 1);
    for entry in entries {
        let entry_path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry_path.is_dir() {
            lines.push(format!("{indent}{name}/"));
            collect_tree(&entry_path, depth + 1, max_depth, lines)?;
        } else {
            lines.push(format!("{indent}{name}"));
        }
    }

    Ok(())
}

fn glob_paths(base_dir: &Path, pattern: &str) -> Result<String> {
    let matcher = GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .with_context(|| format!("invalid glob pattern: {pattern}"))?
        .compile_matcher();

    let mut matches = Vec::new();
    collect_glob_results(base_dir, base_dir, &matcher, &mut matches)?;

    if matches.is_empty() {
        return Ok(format!("GLOB {pattern}\nNO MATCHES"));
    }

    let truncated = matches.len() == MAX_GLOB_MATCHES;
    let mut lines = vec![format!("GLOB {pattern}")];
    lines.extend(matches);
    if truncated {
        lines.push(format!(
            "[stopped at {MAX_GLOB_MATCHES} matches — narrow your pattern]"
        ));
    }

    Ok(truncate_output(lines.join("\n"), pattern))
}

fn collect_glob_results(
    base_dir: &Path,
    path: &Path,
    matcher: &GlobMatcher,
    matches: &mut Vec<String>,
) -> Result<()> {
    if matches.len() >= MAX_GLOB_MATCHES {
        return Ok(());
    }

    let mut entries = fs::read_dir(path)
        .with_context(|| format!("failed to list directory: {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to list directory: {}", path.display()))?;
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let entry_path = entry.path();
        let metadata = fs::metadata(&entry_path)
            .with_context(|| format!("failed to inspect path: {}", entry_path.display()))?;
        let relative = display_relative(base_dir, &entry_path).replace('\\', "/");

        if matcher.is_match(&relative) {
            if metadata.is_dir() {
                matches.push(format!("{relative}/"));
            } else {
                matches.push(relative.clone());
            }
            if matches.len() >= MAX_GLOB_MATCHES {
                return Ok(());
            }
        }

        if metadata.is_dir() {
            collect_glob_results(base_dir, &entry_path, matcher, matches)?;
            if matches.len() >= MAX_GLOB_MATCHES {
                return Ok(());
            }
        }
    }

    Ok(())
}

fn search_path(base_dir: &Path, pattern: &str, path: &str) -> Result<String> {
    let resolved = resolve_path(base_dir, path)?;
    let mut matches = Vec::new();
    collect_search_results(base_dir, &resolved, pattern, &mut matches, 0)?;

    let rel = display_relative(base_dir, &resolved);
    if matches.is_empty() {
        return Ok(format!("SEARCH {pattern} {rel}\nNO MATCHES"));
    }

    let truncated = matches.len() == MAX_SEARCH_MATCHES;
    let mut lines = vec![format!("SEARCH {pattern} {rel}")];
    lines.extend(matches);
    if truncated {
        lines.push(format!(
            "[stopped at {MAX_SEARCH_MATCHES} matches — narrow your pattern or path]"
        ));
    }

    Ok(truncate_output(lines.join("\n"), pattern))
}

fn collect_search_results(
    base_dir: &Path,
    path: &Path,
    pattern: &str,
    matches: &mut Vec<String>,
    depth: usize,
) -> Result<()> {
    if matches.len() >= MAX_SEARCH_MATCHES || depth >= MAX_SEARCH_DEPTH {
        return Ok(());
    }

    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to inspect path: {}", path.display()))?;

    if metadata.is_dir() {
        let mut entries = fs::read_dir(path)
            .with_context(|| format!("failed to list directory: {}", path.display()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("failed to list directory: {}", path.display()))?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            collect_search_results(base_dir, &entry.path(), pattern, matches, depth + 1)?;
            if matches.len() >= MAX_SEARCH_MATCHES {
                return Ok(());
            }
        }
        return Ok(());
    }

    if is_likely_binary(path) {
        return Ok(());
    }

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };

    for (idx, line) in content.lines().enumerate() {
        if line.contains(pattern) {
            matches.push(format!(
                "{}:{}:{}",
                display_relative(base_dir, path),
                idx + 1,
                line
            ));
            if matches.len() >= MAX_SEARCH_MATCHES {
                return Ok(());
            }
        }
    }

    Ok(())
}

fn environment_info(base_dir: &Path) -> Result<String> {
    let cwd = fs::canonicalize(base_dir)
        .with_context(|| format!("failed to resolve base directory: {}", base_dir.display()))?;

    let mut lines = vec![
        "ENV".to_owned(),
        format!("cwd: {}", cwd.display()),
        format!("os: {}", std::env::consts::OS),
        format!("arch: {}", std::env::consts::ARCH),
    ];

    if let Some(shell) = std::env::var_os("SHELL") {
        lines.push(format!("shell: {}", shell.to_string_lossy()));
    }
    if let Some(user) = std::env::var_os("USER") {
        lines.push(format!("user: {}", user.to_string_lossy()));
    }
    if let Some(home) = std::env::var_os("HOME") {
        lines.push(format!("home: {}", home.to_string_lossy()));
    }
    if let Some(path) = std::env::var_os("PATH") {
        lines.push(format!("path: {}", path.to_string_lossy()));
    }

    Ok(truncate_output(lines.join("\n"), "env"))
}

fn is_likely_binary(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some(
            "png"
                | "jpg"
                | "jpeg"
                | "gif"
                | "bmp"
                | "ico"
                | "webp"
                | "exe"
                | "dll"
                | "so"
                | "dylib"
                | "wasm"
                | "pdf"
                | "zip"
                | "tar"
                | "gz"
                | "xz"
                | "bz2"
                | "7z"
                | "lock"
                | "bin"
                | "dat"
                | "db"
                | "sqlite"
        )
    )
}

fn resolve_path(base_dir: &Path, path: &str) -> Result<PathBuf> {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        bail!("absolute paths are not allowed");
    }
    let base_dir = fs::canonicalize(base_dir)
        .with_context(|| format!("failed to resolve base directory: {}", base_dir.display()))?;
    let joined = base_dir.join(candidate);
    let resolved = fs::canonicalize(&joined)
        .with_context(|| format!("path does not exist: {}", joined.display()))?;
    if !resolved.starts_with(&base_dir) {
        bail!("path escapes the workspace");
    }
    Ok(resolved)
}

fn display_relative(base_dir: &Path, path: &Path) -> String {
    path.strip_prefix(base_dir)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn format_system_time(time: std::time::SystemTime) -> Option<String> {
    OffsetDateTime::from(time)
        .to_offset(time::UtcOffset::UTC)
        .format(&Rfc3339)
        .ok()
}

fn format_permissions(metadata: &fs::Metadata) -> String {
    let readonly = metadata.permissions().readonly();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        format!(
            "readonly={readonly}, mode={:o}",
            metadata.permissions().mode() & 0o777
        )
    }
    #[cfg(not(unix))]
    {
        format!("readonly={readonly}")
    }
}

/// Truncate output to MAX_TOOL_OUTPUT_BYTES on a valid UTF-8 char boundary.
fn truncate_output(mut output: String, context: &str) -> String {
    if output.len() <= MAX_TOOL_OUTPUT_BYTES {
        return output;
    }
    let original_len = output.len();
    let boundary = output
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i < MAX_TOOL_OUTPUT_BYTES)
        .last()
        .unwrap_or(0);
    output.truncate(boundary);
    output.push_str(&format!(
        "\n[truncated: {} bytes omitted for '{}' — use a narrower path or pattern]",
        original_len - boundary,
        context,
    ));
    output
}

#[cfg(test)]
mod tests {
    use super::{
        ExtractedToolCall, ToolCall, execute_tool_call, extract_tool_call, parse_tool_call,
    };
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_read_tool_call() {
        let call = parse_tool_call("TOOL: read|src/main.rs").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Read {
                path: "src/main.rs".to_owned()
            }
        );
    }

    #[test]
    fn parses_ls_tool_call() {
        let call = parse_tool_call("TOOL: ls|.").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Ls {
                path: ".".to_owned()
            }
        );
    }

    #[test]
    fn parses_ls_with_trailing_empty_segment() {
        let call = parse_tool_call("TOOL: ls|.|").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Ls {
                path: ".".to_owned()
            }
        );
    }

    #[test]
    fn parses_stat_tool_call() {
        let call = parse_tool_call("TOOL: stat|src/main.rs").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Stat {
                path: "src/main.rs".to_owned()
            }
        );
    }

    #[test]
    fn parses_tree_tool_call() {
        let call = parse_tool_call("TOOL: tree|src|2").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Tree {
                path: "src".to_owned(),
                depth: 2
            }
        );
    }

    #[test]
    fn parses_tree_default_depth() {
        let call = parse_tool_call("TOOL: tree|src").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Tree {
                path: "src".to_owned(),
                depth: 3
            }
        );
    }

    #[test]
    fn parses_glob_tool_call() {
        let call = parse_tool_call("TOOL: glob|**/*.rs").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Glob {
                pattern: "**/*.rs".to_owned()
            }
        );
    }

    #[test]
    fn parses_search_tool_call() {
        let call = parse_tool_call("TOOL: search|prompt|src").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Search {
                pattern: "prompt".to_owned(),
                path: "src".to_owned()
            }
        );
    }

    #[test]
    fn parses_search_with_spaces_in_pattern() {
        let call = parse_tool_call("TOOL: search|my handler fn|src")
            .unwrap()
            .unwrap();
        assert_eq!(
            call,
            ToolCall::Search {
                pattern: "my handler fn".to_owned(),
                path: "src".to_owned()
            }
        );
    }

    #[test]
    fn parses_env_tool_call() {
        let call = parse_tool_call("TOOL: env").unwrap().unwrap();
        assert_eq!(call, ToolCall::Env);
    }

    #[test]
    fn parses_env_with_trailing_empty_segment() {
        let call = parse_tool_call("TOOL: env|").unwrap().unwrap();
        assert_eq!(call, ToolCall::Env);
    }

    #[test]
    fn parses_path_with_spaces() {
        let call = parse_tool_call("TOOL: read|my dir/my file.rs")
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
    fn parses_read_with_whitespace_only_trailing_segment() {
        let call = parse_tool_call("TOOL: read|note.txt|   ").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Read {
                path: "note.txt".to_owned()
            }
        );
    }

    #[test]
    fn ignores_non_tool_output() {
        assert!(parse_tool_call("normal answer").unwrap().is_none());
    }

    #[test]
    fn extracts_first_tool_call_and_preamble_from_multiline_response() {
        let extracted = extract_tool_call("Thinking...\nTOOL: read|README.md\nTOOL: ls|src")
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
        let call = extract_tool_call("I might use TOOL: read|README.md later.").unwrap();
        assert!(call.is_none());
    }

    #[test]
    fn ignores_invalid_tool_lines_when_extracting_tool_call() {
        let call = extract_tool_call("TOOL: nope|README.md\nfinal answer").unwrap();
        assert!(call.is_none());
    }

    #[test]
    fn extracts_tool_call_after_invalid_tool_like_line() {
        let extracted = extract_tool_call("TOOL: nope|README.md\nThinking...\nTOOL: ls|src")
            .unwrap()
            .unwrap();
        assert_eq!(
            extracted,
            ExtractedToolCall {
                preamble: "TOOL: nope|README.md\nThinking...\n".to_owned(),
                call: ToolCall::Ls {
                    path: "src".to_owned()
                }
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
    fn rejects_extra_args_for_env() {
        let err = parse_tool_call("TOOL: env|extra").unwrap_err();
        assert_eq!(err.to_string(), "too many arguments for `env` tool");
    }

    #[test]
    fn rejects_unknown_tool() {
        let err = parse_tool_call("TOOL: write|foo.txt").unwrap_err();
        assert!(err.to_string().contains("unknown tool"));
    }

    #[test]
    fn rejects_too_many_args_for_ls() {
        let err = parse_tool_call("TOOL: ls|src|extra").unwrap_err();
        assert!(err.to_string().contains("too many arguments"));
    }

    #[test]
    fn clamps_tree_depth_to_max() {
        let call = parse_tool_call("TOOL: tree|.|999").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Tree {
                path: ".".to_owned(),
                depth: 8
            }
        );
    }

    #[test]
    fn parses_tree_with_whitespace_only_trailing_segment() {
        let call = parse_tool_call("TOOL: tree|src|2| ").unwrap().unwrap();
        assert_eq!(
            call,
            ToolCall::Tree {
                path: "src".to_owned(),
                depth: 2
            }
        );
    }

    #[test]
    fn executes_read_tool() {
        let root = make_test_dir();
        fs::write(root.join("note.txt"), "hello").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Read {
                path: "note.txt".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(output, "READ note.txt\nhello");
        cleanup(&root);
    }

    #[test]
    fn executes_ls_tool() {
        let root = make_test_dir();
        fs::create_dir(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Ls {
                path: "src".to_owned(),
            },
        )
        .unwrap();
        assert!(output.contains("LS src"));
        assert!(output.contains("FILE main.rs"));
        cleanup(&root);
    }

    #[test]
    fn executes_stat_tool_for_file() {
        let root = make_test_dir();
        fs::write(root.join("note.txt"), "hello").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Stat {
                path: "note.txt".to_owned(),
            },
        )
        .unwrap();
        assert!(output.contains("STAT note.txt"));
        assert!(output.contains("type: FILE"));
        assert!(output.contains("size_bytes: 5"));
        assert!(output.contains("permissions: "));
        cleanup(&root);
    }

    #[test]
    fn executes_stat_tool_for_directory() {
        let root = make_test_dir();
        fs::create_dir(root.join("src")).unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Stat {
                path: "src".to_owned(),
            },
        )
        .unwrap();
        assert!(output.contains("STAT src"));
        assert!(output.contains("type: DIR"));
        cleanup(&root);
    }

    #[test]
    fn executes_tree_tool() {
        let root = make_test_dir();
        fs::create_dir(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("Cargo.toml"), "").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Tree {
                path: ".".to_owned(),
                depth: 2,
            },
        )
        .unwrap();
        assert!(output.contains("TREE"));
        assert!(output.contains("src/"));
        assert!(output.contains("main.rs"));
        assert!(output.contains("Cargo.toml"));
        cleanup(&root);
    }

    #[test]
    fn tree_respects_depth_limit() {
        let root = make_test_dir();
        fs::create_dir_all(root.join("a/b/c")).unwrap();
        fs::write(root.join("a/b/c/deep.rs"), "").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Tree {
                path: ".".to_owned(),
                depth: 2,
            },
        )
        .unwrap();
        assert!(output.contains("b/"));
        assert!(!output.contains("deep.rs"));
        cleanup(&root);
    }

    #[test]
    fn executes_glob_tool() {
        let root = make_test_dir();
        fs::create_dir_all(root.join("src/nested")).unwrap();
        fs::write(root.join("src/main.rs"), "").unwrap();
        fs::write(root.join("src/nested/lib.rs"), "").unwrap();
        fs::write(root.join("Cargo.toml"), "").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Glob {
                pattern: "**/*.rs".to_owned(),
            },
        )
        .unwrap();
        assert!(output.contains("GLOB **/*.rs"));
        assert!(output.contains("src/main.rs"));
        assert!(output.contains("src/nested/lib.rs"));
        assert!(!output.contains("Cargo.toml"));
        cleanup(&root);
    }

    #[test]
    fn glob_includes_hidden_files_and_directories() {
        let root = make_test_dir();
        fs::create_dir_all(root.join(".config")).unwrap();
        fs::write(root.join(".env"), "").unwrap();
        fs::write(root.join(".config/settings.toml"), "").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Glob {
                pattern: "**/.*".to_owned(),
            },
        )
        .unwrap();
        assert!(output.contains(".config/"));
        assert!(output.contains(".env"));
        cleanup(&root);
    }

    #[test]
    fn executes_search_tool() {
        let root = make_test_dir();
        fs::create_dir(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "let prompt = 1;\n").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Search {
                pattern: "prompt".to_owned(),
                path: "src".to_owned(),
            },
        )
        .unwrap();
        assert!(output.contains("SEARCH prompt src"));
        assert!(output.contains("src/main.rs:1:let prompt = 1;"));
        cleanup(&root);
    }

    #[test]
    fn search_skips_binary_extensions() {
        let root = make_test_dir();
        fs::write(root.join("image.png"), "fake binary content with prompt").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Search {
                pattern: "prompt".to_owned(),
                path: ".".to_owned(),
            },
        )
        .unwrap();
        assert!(output.contains("NO MATCHES"));
        cleanup(&root);
    }

    #[test]
    fn search_with_spaces_in_pattern() {
        let root = make_test_dir();
        fs::write(root.join("main.rs"), "let my var = 1;\n").unwrap();
        let output = execute_tool_call(
            &root,
            &ToolCall::Search {
                pattern: "my var".to_owned(),
                path: ".".to_owned(),
            },
        )
        .unwrap();
        assert!(output.contains("main.rs:1:let my var = 1;"));
        cleanup(&root);
    }

    #[test]
    fn executes_env_tool() {
        let root = make_test_dir();
        let output = execute_tool_call(&root, &ToolCall::Env).unwrap();
        assert!(output.contains("ENV"));
        assert!(output.contains("cwd: "));
        assert!(output.contains("os: "));
        assert!(output.contains("arch: "));
        cleanup(&root);
    }

    #[test]
    fn rejects_path_escape() {
        let root = make_test_dir();
        let err = execute_tool_call(
            &root,
            &ToolCall::Read {
                path: "../outside.txt".to_owned(),
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("path"));
        cleanup(&root);
    }

    #[test]
    fn rejects_absolute_path() {
        let root = make_test_dir();
        let err = execute_tool_call(
            &root,
            &ToolCall::Read {
                path: "/etc/passwd".to_owned(),
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("absolute"));
        cleanup(&root);
    }

    fn make_test_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("drover-tools-test-{nanos}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }
}
