use anyhow::{Context, Result, anyhow, bail};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use toml::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub model: Option<String>,
    pub host: Option<String>,
    pub temp: Option<f32>,
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = config_path()?;
        Self::load_from_path(&path)
    }

    fn load_from_path(path: &Path) -> Result<Self> {
        match fs::read_to_string(path) {
            Ok(contents) => Self::parse(&contents),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => Err(err).context(format!("failed to read config file: {}", path.display())),
        }
    }

    fn parse(input: &str) -> Result<Self> {
        let value: Value = toml::from_str(input).context("failed to parse config TOML")?;
        let table = value
            .as_table()
            .context("config root must be a TOML table")?;

        let model = optional_string(table.get("model"), "model")?;
        let host = optional_string(table.get("host"), "host")?;
        let temp = optional_float(table.get("temp"), "temp")?;

        Ok(Self { model, host, temp })
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: Some("llama3".to_owned()),
            host: Some("http://localhost:11434".to_owned()),
            temp: Some(0.7),
        }
    }
}

fn config_path() -> Result<PathBuf> {
    let home = env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home)
        .join(".config")
        .join("drover")
        .join("config.toml"))
}

fn optional_string(value: Option<&Value>, key: &str) -> Result<Option<String>> {
    match value {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => bail!("config key `{key}` must be a string"),
    }
}

fn optional_float(value: Option<&Value>, key: &str) -> Result<Option<f32>> {
    match value {
        None => Ok(None),
        Some(Value::Float(value)) => Ok(Some(*value as f32)),
        Some(Value::Integer(value)) => Ok(Some(*value as f32)),
        Some(_) => bail!("config key `{key}` must be a number"),
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_full_config() {
        let config = Config::parse(
            r#"
model = "llama3"
host = "http://localhost:11434"
temp = 0.7
"#,
        )
        .unwrap();

        assert_eq!(config.model.as_deref(), Some("llama3"));
        assert_eq!(config.host.as_deref(), Some("http://localhost:11434"));
        assert_eq!(config.temp, Some(0.7));
    }

    #[test]
    fn parses_partial_config() {
        let config = Config::parse(
            r#"
host = "http://localhost:11434"
"#,
        )
        .unwrap();

        assert_eq!(config.model, None);
        assert_eq!(config.host.as_deref(), Some("http://localhost:11434"));
        assert_eq!(config.temp, None);
    }

    #[test]
    fn returns_default_when_file_is_missing() {
        let path = unique_test_path("missing.toml");
        let config = Config::load_from_path(&path).unwrap();

        assert_eq!(config, Config::default());
        assert!(!path.exists());
    }

    #[test]
    fn rejects_non_string_model() {
        let err = Config::parse("model = 42").unwrap_err();

        assert_eq!(err.to_string(), "config key `model` must be a string");
    }

    #[test]
    fn rejects_non_numeric_temp() {
        let err = Config::parse(r#"temp = "hot""#).unwrap_err();

        assert_eq!(err.to_string(), "config key `temp` must be a number");
    }

    #[test]
    fn loads_config_from_path() {
        let path = unique_test_path("config.toml");
        let parent = path.parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        fs::write(
            &path,
            r#"
model = "llama3"
host = "http://localhost:11434"
temp = 1
"#,
        )
        .unwrap();

        let config = Config::load_from_path(&path).unwrap();

        assert_eq!(config.model.as_deref(), Some("llama3"));
        assert_eq!(config.host.as_deref(), Some("http://localhost:11434"));
        assert_eq!(config.temp, Some(1.0));

        fs::remove_file(&path).unwrap();
        fs::remove_dir_all(parent).unwrap();
    }

    fn unique_test_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir()
            .join(format!("drover-config-test-{nanos}"))
            .join(name)
    }
}
