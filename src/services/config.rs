use anyhow::{anyhow, bail, Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use toml::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub model: String,
    pub host: String,
    pub temp: f32,
    pub allow_tools: bool,
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

        let mut config = Self::default();

        if let Some(value) = table.get("model") {
            config.model = value
                .as_str()
                .context("config key `model` must be a string")?
                .to_owned();
        }

        if let Some(value) = table.get("host") {
            config.host = value
                .as_str()
                .context("config key `host` must be a string")?
                .to_owned();
        }

        if let Some(value) = table.get("temp") {
            config.temp = match value {
                Value::Float(v) => *v as f32,
                Value::Integer(v) => *v as f32,
                _ => bail!("config key `temp` must be a number"),
            };
        }

        if let Some(value) = table.get("allow_tools") {
            config.allow_tools = value
                .as_bool()
                .context("config key `allow_tools` must be a boolean")?;
        }

        Ok(config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "llama3".to_owned(),
            host: "http://localhost:11434".to_owned(),
            temp: 0.7,
            allow_tools: false,
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

#[cfg(test)]
mod tests {
    use super::Config;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_partial_config() {
        let config = Config::parse(
            r#"
host = "http://localhost:11434"
"#,
        )
        .unwrap();

        assert_eq!(config.model, "llama3");
        assert_eq!(config.host, "http://localhost:11434");
        assert_eq!(config.temp, 0.7);
        assert!(!config.allow_tools);
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
    fn rejects_non_boolean_allow_tools() {
        let err = Config::parse(r#"allow_tools = "yes""#).unwrap_err();
        assert_eq!(err.to_string(), "config key `allow_tools` must be a boolean");
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
allow_tools = true
"#,
        )
        .unwrap();

        let config = Config::load_from_path(&path).unwrap();

        assert_eq!(config.model, "llama3");
        assert_eq!(config.host, "http://localhost:11434");
        assert_eq!(config.temp, 1.0);
        assert!(config.allow_tools);

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
