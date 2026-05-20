use anyhow::{Context, Result};
use minijinja::Environment;
use rllm_core::request::ChatMessage;
use serde::Serialize;

#[derive(Serialize)]
struct TemplateMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct TemplateContext {
    messages: Vec<TemplateMessage>,
    add_generation_prompt: bool,
    bos_token: &'static str,
    eos_token: &'static str,
}

pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = Environment::new();
    env.add_template("chat", template).context("invalid chat template")?;

    let tmpl_messages: Vec<TemplateMessage> = messages
        .iter()
        .map(|m| TemplateMessage { role: m.role.clone(), content: m.content.clone() })
        .collect();

    let ctx = TemplateContext {
        messages: tmpl_messages,
        add_generation_prompt,
        bos_token: "",
        eos_token: "",
    };

    let tmpl = env.get_template("chat")?;
    let rendered = tmpl.render(ctx).context("chat template rendering failed")?;

    Ok(rendered)
}

pub fn render_chat_template_fallback(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> String {
    let mut output = String::new();
    for msg in messages {
        output.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
    }
    if add_generation_prompt {
        output.push_str("<|assistant|>\n");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage { role: "system".into(), content: "You are a helpful assistant.".into() },
            ChatMessage { role: "user".into(), content: "Hello!".into() },
        ]
    }

    #[test]
    fn fallback_renders_messages() {
        let messages = make_messages();
        let result = render_chat_template_fallback(&messages, true);
        assert!(result.contains("system"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("user"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("assistant"));
    }

    #[test]
    fn fallback_without_generation_prompt() {
        let messages = vec![ChatMessage { role: "user".into(), content: "Hello!".into() }];
        let result = render_chat_template_fallback(&messages, false);
        assert!(!result.contains("<|assistant|"));
    }

    #[test]
    fn llama_template_renders() {
        let template = concat!(
            "{% for message in messages %}",
            "{start_header}{{ message.role }}{end_header}\n\n",
            "{{ message.content }}{eot}",
            "{% endfor %}",
            "{% if add_generation_prompt %}",
            "{start_header}assistant{end_header}\n\n",
            "{% endif %}",
        )
        .replace("{start_header}", "<|start_header_id|>")
        .replace("{end_header}", "<|end_header_id|>")
        .replace("{eot}", "<|eot_id|>");

        let messages = make_messages();
        let result = render_chat_template(&template, &messages, true).unwrap();
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn chatml_template_renders() {
        let template = concat!(
            "{% for message in messages %}",
            "{im_start}{{ message.role }}\n{{ message.content }}{im_end}\n",
            "{% endfor %}",
            "{% if add_generation_prompt %}",
            "{im_start}assistant\n",
            "{% endif %}",
        )
        .replace("{im_start}", "<|im_start|>")
        .replace("{im_end}", "<|im_end|>");

        let messages = vec![ChatMessage { role: "user".into(), content: "What is 2+2?".into() }];
        let result = render_chat_template(&template, &messages, true).unwrap();
        assert!(result.contains("user"));
        assert!(result.contains("What is 2+2?"));
        assert!(result.contains("assistant"));
    }

    #[test]
    fn invalid_template_returns_error() {
        let messages = make_messages();
        let result = render_chat_template("{{ unclosed", &messages, false);
        assert!(result.is_err());
    }
}
