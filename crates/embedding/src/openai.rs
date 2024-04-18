use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

#[derive(Debug, Error)]
#[error(
    "\
Error happens at embedding.
INFORMATION: hint = {hint}"
)]
pub struct EmbeddingError {
    pub hint: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: i32,
}

#[derive(Debug, Serialize, Clone)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl EmbeddingRequest {
    pub fn new(model: String, input: String) -> Self {
        Self {
            model,
            input,
            dimensions: None,
            user: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

impl EmbeddingResponse {
    pub fn try_pop_embedding(mut self) -> Result<Vec<f32>, EmbeddingError> {
        match self.data.pop() {
            Some(d) => Ok(d.embedding),
            None => Err(EmbeddingError {
                hint: "no embedding from service".to_string(),
            }),
        }
    }
}

#[derive(Debug, Error)]
#[error(
    "\
Error happens at completion.
INFORMATION: hint = {hint}"
)]
pub struct CompletionError {
    pub hint: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompletionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionData {
    pub object: String,
    pub completion: Vec<f32>,
    pub index: i32,
}

#[derive(Debug, Serialize, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<CompletionMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

impl CompletionRequest {
    pub fn new(model: String, messages: Vec<CompletionMessage>) -> Self {
        Self {
            model,
            messages,
            temperature: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionChoice {
    pub index: i32,
    pub message: CompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub object: String,
    pub choices: Vec<CompletionChoice>,
    pub model: String,
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn try_pop_completion(mut self) -> Result<String, CompletionError> {
        match self.choices.pop() {
            Some(d) => Ok(d.message.content),
            None => Err(CompletionError {
                hint: "no completion from service".to_string(),
            }),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

mod tests {

    #[test]
    fn test_completion_request() {
        use super::*;
        let req = CompletionRequest {
            model: "moonshot-v1-8k".to_string(),
            messages: vec![
                CompletionMessage {
                    role: "system".to_string(),
                    content: "you are kimi.".to_string(),
                },
                CompletionMessage {
                    role: "user".to_string(),
                    content: "hello, 1+1=?".to_string(),
                },
            ],
            temperature: Some(0.3),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"model":"moonshot-v1-8k","messages":[{"role":"system","content":"you are kimi."},{"role":"user","content":"hello, 1+1=?"}],"temperature":0.3}"#
        );
    }

    #[test]
    fn test_completion_response() {
        use super::*;
        let json = r#"
		{"id":"cmpl-870129f39e2e4a0a836a47c3ff1c18a3","object":"chat.completion","created":817482,"model":"moonshot-v1-8k","choices":[{"index":0,"message":{"role":"assistant","content":"1+1=2"},"finish_reason":"stop"}],"usage":{"prompt_tokens":51,"completion_tokens":6,"total_tokens":57}}"#;
        let resp: CompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content, "1+1=2");
    }
}
