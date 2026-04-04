use anyhow::{Context, Result, anyhow, bail};
use json::{JsonValue, object};
use std::io::{BufRead, BufReader, Read};

#[derive(Debug, Clone, PartialEq)]
pub struct OllamaClient {
    host: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    pub temp: Option<f32>,
    pub stream: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateResponse {
    pub response: String,
    pub done: bool,
    pub model: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub eval_count: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateChunk {
    pub response: String,
    pub done: bool,
    pub model: Option<String>,
}

impl OllamaClient {
    pub fn new(host: impl Into<String>) -> Self {
        Self {
            host: normalize_host(host.into()),
        }
    }

    pub fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let response = self
            .send_generate_request(request)
            .context("failed to call Ollama generate endpoint")?;
        let body = read_response_body(response)?;
        parse_generate_response(&body)
    }

    pub fn generate_streaming<F>(
        &self,
        request: &GenerateRequest,
        mut on_chunk: F,
    ) -> Result<GenerateResponse>
    where
        F: FnMut(&GenerateChunk) -> Result<()>,
    {
        let response = self
            .send_generate_request(request)
            .context("failed to call Ollama generate endpoint")?;

        let mut reader = BufReader::new(response.into_reader());
        let mut line = String::new();
        let mut accumulated = String::new();
        let mut final_response = None;

        loop {
            line.clear();
            let bytes_read = reader
                .read_line(&mut line)
                .context("failed to read Ollama streaming response")?;
            if bytes_read == 0 {
                break;
            }

            if line.trim().is_empty() {
                continue;
            }

            let parsed = parse_generate_json_line(line.trim_end())?;
            if !parsed.chunk.done && parsed.chunk.response.is_empty() {
                bail!("missing response content in Ollama stream chunk");
            }

            accumulated.push_str(&parsed.chunk.response);
            on_chunk(&parsed.chunk)?;

            if parsed.chunk.done {
                final_response = Some(parsed.response.with_response(accumulated.clone()));
                break;
            }
        }

        final_response.ok_or_else(|| anyhow!("Ollama stream ended before a final done chunk"))
    }

    fn send_generate_request(&self, request: &GenerateRequest) -> Result<ureq::Response> {
        let url = self.generate_url();
        let body = build_generate_body(request).dump();

        match ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body)
        {
            Ok(response) => Ok(response),
            Err(ureq::Error::Status(code, response)) => {
                let body = read_response_body(response)
                    .unwrap_or_else(|_| String::from("<unreadable response body>"));
                Err(anyhow!("Ollama request failed with status {code}: {body}"))
            }
            Err(ureq::Error::Transport(err)) => Err(anyhow!(err))
                .context(format!("failed to connect to Ollama at {}", self.host)),
        }
    }

    fn generate_url(&self) -> String {
        format!("{}/api/generate", self.host)
    }
}

impl GenerateResponse {
    fn with_response(mut self, response: String) -> Self {
        self.response = response;
        self
    }
}

fn normalize_host(host: String) -> String {
    host.trim_end_matches('/').to_owned()
}

fn build_generate_body(request: &GenerateRequest) -> JsonValue {
    let mut body = object! {
        model: request.model.as_str(),
        prompt: request.prompt.as_str(),
        stream: request.stream,
    };

    if let Some(system) = request.system.as_deref() {
        body["system"] = system.into();
    }

    if let Some(temp) = request.temp {
        body["options"] = object! { temperature: temp };
    }

    body
}

fn read_response_body(response: ureq::Response) -> Result<String> {
    let mut body = String::new();
    let mut reader = response.into_reader();
    reader
        .read_to_string(&mut body)
        .context("failed to read Ollama response body")?;
    Ok(body)
}

fn parse_generate_response(body: &str) -> Result<GenerateResponse> {
    let json = json::parse(body).context("failed to parse Ollama JSON response")?;
    parse_generate_json(&json)
}

fn parse_generate_json_line(line: &str) -> Result<ParsedChunk> {
    let json = json::parse(line).context("failed to parse Ollama streaming JSON")?;
    let response = parse_generate_json(&json)?;
    let chunk = GenerateChunk {
        response: required_string(&json, "response")?.to_owned(),
        done: required_bool(&json, "done")?,
        model: optional_string(&json, "model"),
    };

    Ok(ParsedChunk { chunk, response })
}

fn parse_generate_json(json: &JsonValue) -> Result<GenerateResponse> {
    Ok(GenerateResponse {
        response: required_string(json, "response")?.to_owned(),
        done: required_bool(json, "done")?,
        model: optional_string(json, "model"),
        total_duration: optional_u64(json, "total_duration")?,
        load_duration: optional_u64(json, "load_duration")?,
        prompt_eval_count: optional_u64(json, "prompt_eval_count")?,
        eval_count: optional_u64(json, "eval_count")?,
    })
}

fn required_string<'a>(json: &'a JsonValue, key: &str) -> Result<&'a str> {
    json[key]
        .as_str()
        .ok_or_else(|| anyhow!("missing or invalid `{key}` in Ollama response"))
}

fn required_bool(json: &JsonValue, key: &str) -> Result<bool> {
    json[key]
        .as_bool()
        .ok_or_else(|| anyhow!("missing or invalid `{key}` in Ollama response"))
}

fn optional_string(json: &JsonValue, key: &str) -> Option<String> {
    json[key].as_str().map(str::to_owned)
}

fn optional_u64(json: &JsonValue, key: &str) -> Result<Option<u64>> {
    if json[key].is_null() {
        return Ok(None);
    }

    let value = json[key]
        .as_u64()
        .ok_or_else(|| anyhow!("invalid `{key}` in Ollama response"))?;
    Ok(Some(value))
}

#[derive(Debug)]
struct ParsedChunk {
    chunk: GenerateChunk,
    response: GenerateResponse,
}

#[cfg(test)]
mod tests {
    use super::{
        GenerateRequest, build_generate_body, normalize_host, parse_generate_json_line,
        parse_generate_response,
    };

    #[test]
    fn build_generate_body_omits_optional_fields_when_absent() {
        let request = GenerateRequest {
            model: "llama3".to_owned(),
            prompt: "write a haiku".to_owned(),
            system: None,
            temp: None,
            stream: false,
        };

        let body = build_generate_body(&request);

        assert_eq!(body["model"].as_str(), Some("llama3"));
        assert_eq!(body["prompt"].as_str(), Some("write a haiku"));
        assert_eq!(body["stream"].as_bool(), Some(false));
        assert!(body["system"].is_null());
        assert!(body["options"].is_null());
    }

    #[test]
    fn build_generate_body_includes_system_and_temperature() {
        let request = GenerateRequest {
            model: "llama3".to_owned(),
            prompt: "write a sonnet".to_owned(),
            system: Some("you are a poet".to_owned()),
            temp: Some(0.7),
            stream: true,
        };

        let body = build_generate_body(&request);

        assert_eq!(body["system"].as_str(), Some("you are a poet"));
        let temperature = body["options"]["temperature"].as_f32().unwrap();
        assert!((temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(body["stream"].as_bool(), Some(true));
    }

    #[test]
    fn normalizes_host_trailing_slash() {
        assert_eq!(
            normalize_host("http://localhost:11434/".to_owned()),
            "http://localhost:11434"
        );
    }

    #[test]
    fn parses_non_streaming_response() {
        let response = parse_generate_response(
            r#"{
  "model": "llama3",
  "response": "hello",
  "done": true,
  "total_duration": 10,
  "load_duration": 2,
  "prompt_eval_count": 3,
  "eval_count": 4
}"#,
        )
        .unwrap();

        assert_eq!(response.response, "hello");
        assert!(response.done);
        assert_eq!(response.model.as_deref(), Some("llama3"));
        assert_eq!(response.total_duration, Some(10));
        assert_eq!(response.load_duration, Some(2));
        assert_eq!(response.prompt_eval_count, Some(3));
        assert_eq!(response.eval_count, Some(4));
    }

    #[test]
    fn parses_streaming_chunk() {
        let parsed =
            parse_generate_json_line(r#"{"model":"llama3","response":"hel","done":false}"#)
                .unwrap();

        assert_eq!(parsed.chunk.response, "hel");
        assert!(!parsed.chunk.done);
        assert_eq!(parsed.chunk.model.as_deref(), Some("llama3"));
    }

    #[test]
    fn parses_final_streaming_chunk_metadata() {
        let parsed = parse_generate_json_line(
            r#"{
  "model": "llama3",
  "response": "",
  "done": true,
  "total_duration": 10,
  "load_duration": 2,
  "prompt_eval_count": 3,
  "eval_count": 4
}"#,
        )
        .unwrap();

        assert!(parsed.chunk.done);
        assert_eq!(parsed.response.total_duration, Some(10));
        assert_eq!(parsed.response.eval_count, Some(4));
    }

    #[test]
    fn rejects_malformed_streaming_json() {
        let err = parse_generate_json_line("{").unwrap_err();

        assert_eq!(err.to_string(), "failed to parse Ollama streaming JSON");
    }

    #[test]
    fn rejects_missing_response_field() {
        let err = parse_generate_response(r#"{"done":true}"#).unwrap_err();

        assert_eq!(err.to_string(), "missing or invalid `response` in Ollama response");
    }

    #[test]
    fn rejects_invalid_numeric_metadata() {
        let err = parse_generate_response(
            r#"{"response":"hello","done":true,"total_duration":"fast"}"#,
        )
        .unwrap_err();

        assert_eq!(err.to_string(), "invalid `total_duration` in Ollama response");
    }
}
