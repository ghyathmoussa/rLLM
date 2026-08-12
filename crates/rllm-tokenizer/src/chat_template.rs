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
    messages: Vec<serde_json::Value>,
    tools: Option<Vec<serde_json::Value>>,
    tool_choice: Option<serde_json::Value>,
    parallel_tool_calls: bool,
    add_generation_prompt: bool,
    bos_token: &'static str,
    eos_token: &'static str,
}

pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    let tmpl_messages: Vec<TemplateMessage> = messages
        .iter()
        .map(|m| TemplateMessage { role: m.role.clone(), content: m.content.clone() })
        .collect();

    let messages = tmpl_messages
        .into_iter()
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()
        .context("serializing chat messages")?;
    render_chat_template_with_tools(template, messages, None, None, true, add_generation_prompt)
}

pub fn render_chat_template_with_tools(
    template: &str,
    messages: Vec<serde_json::Value>,
    tools: Option<Vec<serde_json::Value>>,
    tool_choice: Option<serde_json::Value>,
    parallel_tool_calls: bool,
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = Environment::new();
    env.add_template("chat", template).context("invalid chat template")?;

    let ctx = TemplateContext {
        messages,
        tools,
        tool_choice,
        parallel_tool_calls,
        add_generation_prompt,
        bos_token: "",
        eos_token: "",
    };

    let tmpl = env.get_template("chat")?;
    let rendered = tmpl.render(ctx).context("chat template rendering failed")?;

    Ok(rendered)
}

pub fn render_chat_template_fallback_with_tools(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    add_generation_prompt: bool,
) -> Result<String> {
    let mut output = String::new();
    if let Some(tools) = tools {
        output.push_str("<|system|>\nYou have access to the following functions. ");
        output.push_str(
            "When a function is needed, respond with a JSON object containing its name and arguments.\n",
        );
        output.push_str(
            &serde_json::to_string_pretty(tools)
                .context("serializing tools for fallback prompt")?,
        );
        output.push('\n');
    }
    for message in messages {
        let role = message.get("role").and_then(serde_json::Value::as_str).unwrap_or("user");
        let content = message.get("content").and_then(serde_json::Value::as_str).unwrap_or("");
        output.push_str(&format!("<|{role}|>\n{content}\n"));
    }
    if add_generation_prompt {
        output.push_str("<|assistant|>\n");
    }
    Ok(output)
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

    #[test]
    fn tool_context_is_available_to_chat_template() {
        let template = "{{ tools | tojson }}|{{ tool_choice }}|{{ parallel_tool_calls }}";
        let messages = vec![serde_json::json!({"role": "user", "content": "weather"})];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}}
        })];
        let rendered = render_chat_template_with_tools(
            template,
            messages,
            Some(tools),
            Some(serde_json::json!("required")),
            false,
            true,
        )
        .unwrap();
        assert!(rendered.contains("get_weather"));
        assert!(rendered.contains("required"));
        assert!(rendered.ends_with("false"));
    }
}
