use anyhow::Result;

pub fn render_chat_template(
    _template: &str,
    messages: &[(String, String)],
    add_generation_prompt: bool,
) -> Result<String> {
    // Placeholder: simple concatenation for Llama-style messages.
    // Full Jinja2 template rendering will be added in Phase 3.
    let mut output = String::new();
    for (role, content) in messages {
        output.push_str(&format!("<|{role}|>\n{content}\n"));
    }
    if add_generation_prompt {
        output.push_str("<|assistant|]\n");
    }
    Ok(output)
}
