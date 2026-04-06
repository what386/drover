use crate::tools::ToolCall;
use anyhow::{Context, Result, bail};
use globset::{GlobBuilder, GlobMatcher};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use walkdir::WalkDir;

const MAX_TOOL_OUTPUT_BYTES: usize = 16 * 1024;
const MAX_GLOB_MATCHES: usize = 500;
const MAX_SEARCH_MATCHES: usize = 200;
const MAX_SEARCH_DEPTH: usize = 8;

pub fn execute_tool_call(base_dir: &Path, call: &ToolCall) -> Result<String> {
    match call {
        ToolCall::Read { path } => read_file(base_dir, path),
        ToolCall::List { path } => list_path(base_dir, path),
        ToolCall::Stat { path } => stat_path(base_dir, path),
        ToolCall::Glob { pattern, exclude } => glob_paths(base_dir, pattern, exclude.as_deref()),
        ToolCall::Search { pattern, path } => search_path(base_dir, pattern, path),
    }
}

fn read_file(base_dir: &Path, path: &str) -> Result<String> {
    let resolved = resolve_path(base_dir, path)?;
    let content = fs::read_to_string(&resolved)
        .context(format!("failed to read file: {}", resolved.display()))?;
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
        .context(format!("failed to inspect path: {}", resolved.display()))?;

    if metadata.is_file() {
        return Ok(format!(
            "LS {}\nFILE",
            display_relative(base_dir, &resolved)
        ));
    }

    let mut lines = vec![format!("LS {}", display_relative(base_dir, &resolved))];
    for (name, kind) in read_dir_rows(&resolved)? {
        lines.push(format!("{kind} {name}"));
    }

    Ok(truncate_output(lines.join("\n"), path))
}

fn stat_path(base_dir: &Path, path: &str) -> Result<String> {
    let resolved = resolve_path(base_dir, path)?;
    let metadata = fs::metadata(&resolved)
        .context(format!("failed to inspect path: {}", resolved.display()))?;
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

fn glob_paths(base_dir: &Path, pattern: &str, exclude: Option<&str>) -> Result<String> {
    let matcher = compile_glob_matcher(pattern, "pattern")?;
    let exclude_matcher = exclude
        .map(|exclude| compile_glob_matcher(exclude, "exclude pattern"))
        .transpose()?;

    let mut matches = Vec::new();
    let mut walker = WalkDir::new(base_dir)
        .follow_links(false)
        .sort_by_file_name()
        .min_depth(1)
        .into_iter();

    while let Some(entry) = walker.next() {
        let entry = entry.context(format!("failed to walk directory: {}", base_dir.display()))?;
        let relative = normalized_relative(base_dir, entry.path());

        if exclude_matcher
            .as_ref()
            .is_some_and(|exclude| exclude.is_match(&relative))
        {
            if entry.file_type().is_dir() {
                walker.skip_current_dir();
            }
            continue;
        }

        if matcher.is_match(&relative) {
            if path_displays_as_dir(entry.path(), entry.file_type()) {
                matches.push(format!("{relative}/"));
            } else {
                matches.push(relative);
            }
        }

        if matches.len() >= MAX_GLOB_MATCHES {
            break;
        }
    }

    let glob_description = format_glob_description(pattern, exclude);
    let truncated = matches.len() == MAX_GLOB_MATCHES;
    Ok(format_match_output(
        "GLOB",
        &glob_description,
        matches,
        truncated,
        format!("[stopped at {MAX_GLOB_MATCHES} matches — narrow your pattern]"),
        &glob_description,
    ))
}

fn compile_glob_matcher(pattern: &str, label: &str) -> Result<GlobMatcher> {
    Ok(GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .context(format!("invalid glob {label}: {pattern}"))?
        .compile_matcher())
}

fn format_glob_description(pattern: &str, exclude: Option<&str>) -> String {
    match exclude {
        Some(exclude) => format!("{pattern} EXCLUDE {exclude}"),
        None => pattern.to_owned(),
    }
}

fn search_path(base_dir: &Path, pattern: &str, path: &str) -> Result<String> {
    let resolved = resolve_path(base_dir, path)?;
    let mut matches = Vec::new();
    let rel = display_relative(base_dir, &resolved);
    let metadata = fs::metadata(&resolved)
        .context(format!("failed to inspect path: {}", resolved.display()))?;

    if metadata.is_dir() {
        let walker = WalkDir::new(&resolved)
            .follow_links(false)
            .sort_by_file_name()
            .min_depth(1)
            .max_depth(MAX_SEARCH_DEPTH)
            .into_iter();

        for entry in walker {
            let entry =
                entry.context(format!("failed to walk directory: {}", resolved.display()))?;
            let file_type = entry.file_type();

            if file_type.is_symlink() || !file_type.is_file() {
                continue;
            }

            push_search_matches(base_dir, entry.path(), pattern, &mut matches)?;
            if matches.len() >= MAX_SEARCH_MATCHES {
                break;
            }
        }
    } else {
        push_search_matches(base_dir, &resolved, pattern, &mut matches)?;
    }

    let truncated = matches.len() == MAX_SEARCH_MATCHES;
    Ok(format_match_output(
        "SEARCH",
        &format!("{pattern} {rel}"),
        matches,
        truncated,
        format!("[stopped at {MAX_SEARCH_MATCHES} matches — narrow your pattern or path]"),
        pattern,
    ))
}

fn is_likely_binary(path: &Path) -> bool {
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return false,
    };

    let mut buffer = [0_u8; 500];
    match file.read(&mut buffer) {
        Ok(bytes_read) => buffer[..bytes_read].contains(&0),
        Err(_) => false,
    }
}

fn push_search_matches(
    base_dir: &Path,
    path: &Path,
    pattern: &str,
    matches: &mut Vec<String>,
) -> Result<()> {
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
                break;
            }
        }
    }

    Ok(())
}

fn read_dir_rows(path: &Path) -> Result<Vec<(String, &'static str)>> {
    let mut rows = fs::read_dir(path)
        .context(format!("failed to list directory: {}", path.display()))?
        .map(|entry| build_dir_row(entry, path))
        .collect::<Result<Vec<_>>>()?;
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(rows)
}

fn build_dir_row(entry: std::io::Result<fs::DirEntry>, parent: &Path) -> Result<(String, &'static str)> {
    let entry = entry.context(format!("failed to list directory: {}", parent.display()))?;
    let path = entry.path();
    let file_type = entry
        .file_type()
        .context(format!("failed to inspect path: {}", path.display()))?;
    let name = entry.file_name().to_string_lossy().into_owned();
    let kind = if path_displays_as_dir(&path, file_type) {
        "DIR"
    } else {
        "FILE"
    };
    Ok((name, kind))
}

fn path_displays_as_dir(path: &Path, file_type: std::fs::FileType) -> bool {
    if file_type.is_dir() {
        return true;
    }

    file_type.is_symlink() && fs::metadata(path).map(|metadata| metadata.is_dir()).unwrap_or(false)
}

fn normalized_relative(base_dir: &Path, path: &Path) -> String {
    display_relative(base_dir, path).replace('\\', "/")
}

fn format_match_output(
    label: &str,
    description: &str,
    matches: Vec<String>,
    truncated: bool,
    truncated_notice: String,
    truncate_context: &str,
) -> String {
    let mut lines = vec![format!("{label} {description}")];

    if matches.is_empty() {
        lines.push("NO MATCHES".to_owned());
    } else {
        lines.extend(matches);
        if truncated {
            lines.push(truncated_notice);
        }
    }

    truncate_output(lines.join("\n"), truncate_context)
}

fn resolve_path(base_dir: &Path, path: &str) -> Result<PathBuf> {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        bail!("absolute paths are not allowed");
    }
    let base_dir = fs::canonicalize(base_dir).context(format!(
        "failed to resolve base directory: {}",
        base_dir.display()
    ))?;
    let joined = base_dir.join(candidate);
    let resolved =
        fs::canonicalize(&joined).context(format!("path does not exist: {}", joined.display()))?;
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
    use super::execute_tool_call;
    use crate::tools::ToolCall;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

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
                exclude: None,
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
                exclude: None,
            },
        )
        .unwrap();
        assert!(output.contains(".config/"));
        assert!(output.contains(".env"));
        cleanup(&root);
    }

    #[test]
    fn executes_glob_tool_with_exclude() {
        let root = make_test_dir();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("target/generated")).unwrap();
        fs::write(root.join("src/main.rs"), "").unwrap();
        fs::write(root.join("target/generated/lib.rs"), "").unwrap();

        let output = execute_tool_call(
            &root,
            &ToolCall::Glob {
                pattern: "**/*.rs".to_owned(),
                exclude: Some("target/**/*".to_owned()),
            },
        )
        .unwrap();

        assert!(output.contains("GLOB **/*.rs EXCLUDE target/**/*"));
        assert!(output.contains("src/main.rs"));
        assert!(!output.contains("target/generated/lib.rs"));
        cleanup(&root);
    }

    #[test]
    fn glob_prunes_excluded_directories() {
        let root = make_test_dir();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("target/deep/nested")).unwrap();
        fs::write(root.join("src/main.rs"), "").unwrap();
        fs::write(root.join("target/deep/nested/generated.rs"), "").unwrap();

        let output = execute_tool_call(
            &root,
            &ToolCall::Glob {
                pattern: "**".to_owned(),
                exclude: Some("target/**/*".to_owned()),
            },
        )
        .unwrap();

        assert!(output.contains("src/"));
        assert!(output.contains("src/main.rs"));
        assert!(!output.contains("target/deep/"));
        assert!(!output.contains("generated.rs"));
        cleanup(&root);
    }

    #[test]
    fn search_skips_binary_extensions() {
        let root = make_test_dir();
        fs::write(root.join("image.png"), b"fake\0binary content with prompt").unwrap();
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
    fn search_respects_inclusive_depth_limit() {
        let root = make_test_dir();
        let within_limit = root.join("1/2/3/4/5/6/7");
        let beyond_limit = within_limit.join("8");
        fs::create_dir_all(&within_limit).unwrap();
        fs::create_dir_all(&beyond_limit).unwrap();
        fs::write(within_limit.join("hit.rs"), "needle\n").unwrap();
        fs::write(beyond_limit.join("miss.rs"), "needle\n").unwrap();

        let output = execute_tool_call(
            &root,
            &ToolCall::Search {
                pattern: "needle".to_owned(),
                path: ".".to_owned(),
            },
        )
        .unwrap();

        assert!(output.contains("1/2/3/4/5/6/7/hit.rs:1:needle"));
        assert!(!output.contains("1/2/3/4/5/6/7/8/miss.rs:1:needle"));
        cleanup(&root);
    }

    #[cfg(unix)]
    #[test]
    fn glob_matches_symlink_path_without_descending_into_target() {
        let root = make_test_dir();
        let outside = make_test_dir();
        fs::create_dir_all(outside.join("nested")).unwrap();
        fs::write(outside.join("nested/secret.rs"), "").unwrap();
        symlink(&outside, root.join("linked-src")).unwrap();

        let symlink_match = execute_tool_call(
            &root,
            &ToolCall::Glob {
                pattern: "**/linked-src".to_owned(),
                exclude: None,
            },
        )
        .unwrap();
        assert!(symlink_match.contains("linked-src/"));

        let rs_match = execute_tool_call(
            &root,
            &ToolCall::Glob {
                pattern: "**/*.rs".to_owned(),
                exclude: None,
            },
        )
        .unwrap();
        assert!(!rs_match.contains("secret.rs"));

        cleanup(&root);
        cleanup(&outside);
    }

    #[cfg(unix)]
    #[test]
    fn search_does_not_follow_symlinked_directories() {
        let root = make_test_dir();
        let outside = make_test_dir();
        fs::create_dir_all(outside.join("nested")).unwrap();
        fs::write(outside.join("nested/secret.rs"), "needle\n").unwrap();
        symlink(&outside, root.join("linked")).unwrap();

        let output = execute_tool_call(
            &root,
            &ToolCall::Search {
                pattern: "needle".to_owned(),
                path: ".".to_owned(),
            },
        )
        .unwrap();

        assert!(output.contains("NO MATCHES"));

        cleanup(&root);
        cleanup(&outside);
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
