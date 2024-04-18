use crate::datatype::memory_vecf32::Vecf32Output;
use crate::gucs::embedding::openai_options;
use base::scalar::*;
use base::vector::*;
use embedding::openai::CompletionMessage;
use embedding::openai_completion;
use embedding::openai_embedding;
use pgrx::error;

#[pgrx::pg_extern(volatile, strict, parallel_safe)]
fn _vectors_text2vec_openai(input: String, model: String) -> Vecf32Output {
    let options = openai_options();
    let resp = match openai_embedding(input, model, options) {
        Ok(r) => r,
        Err(e) => error!("{}", e.to_string()),
    };
    let embedding = match resp.try_pop_embedding() {
        Ok(emb) => emb.into_iter().map(F32).collect::<Vec<_>>(),
        Err(e) => error!("{}", e.to_string()),
    };

    Vecf32Output::new(Vecf32Borrowed::new(&embedding))
}

#[pgrx::pg_extern(volatile, strict, parallel_safe)]
fn _vectors_extract_pdf(input: &[u8], model: String, prompt: String) -> String {
    let text = match pdf_extract::extract_text_from_mem(input) {
        Ok(text) => text,
        Err(e) => error!(
            "Error happens at completion.\nINFORMATION: hint = {}",
            e.to_string()
        ),
    };
    let options = openai_options();
    let resp = match openai_completion(
        vec![
            CompletionMessage {
                role: "user".to_string(),
                content: text,
            },
            CompletionMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ],
        model,
        options,
    ) {
        Ok(r) => r,
        Err(e) => error!("{}", e.to_string()),
    };
    match resp.try_pop_completion() {
        Ok(c) => c,
        Err(e) => error!("{}", e.to_string()),
    }
}
