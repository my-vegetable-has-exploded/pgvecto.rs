pub mod openai;

use crate::openai::EmbeddingError;
use crate::openai::EmbeddingRequest;
use crate::openai::EmbeddingResponse;
use openai::CompletionError;
use openai::CompletionMessage;
use openai::CompletionRequest;
use openai::CompletionResponse;
use reqwest::blocking::Client;
use std::time::Duration;

pub struct OpenAIOptions {
    pub base_url: String,
    pub api_key: String,
}

pub fn openai_embedding(
    input: String,
    model: String,
    opt: OpenAIOptions,
) -> Result<EmbeddingResponse, EmbeddingError> {
    let url = format!("{}/embeddings", opt.base_url);
    let client = match Client::builder().timeout(Duration::from_secs(30)).build() {
        Ok(c) => c,
        Err(e) => {
            return Err(EmbeddingError {
                hint: e.to_string(),
            })
        }
    };
    let form: EmbeddingRequest = EmbeddingRequest::new(model.to_string(), input);
    let resp = match client
        .post(url)
        .header("Authorization", format!("Bearer {}", opt.api_key))
        .form(&form)
        .send()
    {
        Ok(c) => c,
        Err(e) => {
            return Err(EmbeddingError {
                hint: e.to_string(),
            })
        }
    };
    match resp.json::<EmbeddingResponse>() {
        Ok(c) => Ok(c),
        Err(e) => Err(EmbeddingError {
            hint: e.to_string(),
        }),
    }
}

pub fn openai_completion(
    messages: Vec<CompletionMessage>,
    model: String,
    opt: OpenAIOptions,
) -> Result<CompletionResponse, CompletionError> {
    let url = format!("{}/chat/completions", opt.base_url);
    let client = match Client::builder().timeout(Duration::from_secs(30)).build() {
        Ok(c) => c,
        Err(e) => {
            return Err(CompletionError {
                hint: e.to_string(),
            })
        }
    };
    let form: CompletionRequest = CompletionRequest::new(model.to_string(), messages);
    let builder = client
        .post(url)
        .header("Authorization", format!("Bearer {}", opt.api_key));

    let builder = builder.json(&form);
    let resp = match builder.send() {
        Ok(c) => c,
        Err(e) => {
            return Err(CompletionError {
                hint: e.to_string(),
            })
        }
    };
    match resp.json::<CompletionResponse>() {
        Ok(c) => Ok(c),
        Err(e) => Err(CompletionError {
            hint: e.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::OpenAIOptions;
    use super::{openai_completion, openai_embedding};
    use crate::openai::{
        CompletionChoice, CompletionMessage, CompletionResponse, EmbeddingData, EmbeddingResponse,
        Usage,
    };
    use httpmock::Method::POST;
    use httpmock::MockServer;

    fn mock_embedding_server(resp: EmbeddingResponse) -> MockServer {
        let server = MockServer::start();
        let data = serde_json::to_string(&resp).unwrap();
        let _ = server.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .header("content-type", "text/html; charset=UTF-8")
                .body(data);
        });
        server
    }

    fn mock_completion_server(resp: CompletionResponse) -> MockServer {
        let server = MockServer::start();
        let data = serde_json::to_string(&resp).unwrap();
        let _ = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "text/html; charset=UTF-8")
                .body(data);
        });
        server
    }

    #[test]
    fn test_openai_embedding_successful() {
        let embedding = vec![1.0, 2.0, 3.0];
        let resp = EmbeddingResponse {
            object: "mock-object".to_string(),
            data: vec![EmbeddingData {
                object: "mock-object".to_string(),
                embedding: embedding.clone(),
                index: 0,
            }],
            model: "mock-model".to_string(),
            usage: Usage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        };
        let server = mock_embedding_server(resp);

        let opt = OpenAIOptions {
            base_url: server.url(""),
            api_key: "fake-key".to_string(),
        };

        let real_resp = openai_embedding("mock-input".to_string(), "mock-model".to_string(), opt);
        assert!(real_resp.is_ok());
        let real_embedding = real_resp.unwrap().try_pop_embedding();
        assert!(real_embedding.is_ok());
    }

    #[test]
    fn test_openai_embedding_empty_embedding() {
        let resp = EmbeddingResponse {
            object: "mock-object".to_string(),
            data: vec![],
            model: "mock-model".to_string(),
            usage: Usage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        };
        let server = mock_embedding_server(resp);

        let opt = OpenAIOptions {
            base_url: server.url(""),
            api_key: "fake-key".to_string(),
        };

        let real_resp = openai_embedding("mock-input".to_string(), "mock-model".to_string(), opt);
        assert!(real_resp.is_ok());
        let real_embedding = real_resp.unwrap().try_pop_embedding();
        assert!(real_embedding.is_err());
    }

    #[test]
    fn test_openai_embedding_error() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(502)
                .header("content-type", "text/html; charset=UTF-8")
                .body("502 Bad Gateway");
        });

        let opt = OpenAIOptions {
            base_url: server.url(""),
            api_key: "fake-key".to_string(),
        };

        let real_resp = openai_embedding("mock-input".to_string(), "mock-model".to_string(), opt);
        assert!(real_resp.is_err());
    }

    #[test]
    fn test_openai_completion_successful() {
        let completion = "mock-completion".to_string();
        let resp = CompletionResponse {
            object: "mock-object".to_string(),
            choices: vec![CompletionChoice {
                index: 0,
                message: CompletionMessage {
                    role: "mock-role".to_string(),
                    content: completion.clone(),
                },
                finish_reason: "mock-reason".to_string(),
            }],
            model: "mock-model".to_string(),
            usage: Usage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        };
        let server = mock_completion_server(resp);

        let opt = OpenAIOptions {
            base_url: server.url(""),
            api_key: "fake-key".to_string(),
        };

        let real_resp = openai_completion(
            vec![CompletionMessage {
                role: "mock-role".to_string(),
                content: "mock-content".to_string(),
            }],
            "mock-model".to_string(),
            opt,
        );
        assert!(real_resp.is_ok());
        let real_completion = real_resp.unwrap().try_pop_completion();
        assert!(real_completion.is_ok());
    }
}
