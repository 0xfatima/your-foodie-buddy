# Gradio UI for Connoisseur Companion.
# Uses chat_with_agent() from agent.py, so the agent logic stays in one place.
import gradio as gr
from agent import chat_with_agent

QUICK_PROMPTS = [
    ("Find moody restaurants", "Find me some moody restaurants"),
    ("Tell me about Iron & Embers", "Tell me about Iron & Embers"),
    ("Zen dining in Little Tokyo?", "What's a zen dining experience in Little Tokyo?"),
    ("Read a review", "What do reviewers say about Sakura Garden?"),
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Young+Serif&family=Figtree:wght@400;500;600&display=swap');

:root {
  --cc-olive: #34402A;
  --cc-saffron: #E3A53C;
  --cc-sage: #EEF2E6;
}

.gradio-container { font-family: 'Figtree', system-ui, sans-serif !important; max-width: 1100px !important; margin: 0 auto !important; }

#cc-header {
  background: var(--cc-olive);
  color: #F7F8F2;
  border-radius: 14px;
  padding: 28px 32px;
  border-bottom: 5px solid var(--cc-saffron);
}
#cc-header h1 { font-family: 'Young Serif', Georgia, serif; font-size: 2.4rem; font-weight: 400; margin: 0 0 6px; color: #F7F8F2; }
#cc-header p { margin: 0; font-size: 1.05rem; color: #D9DFCC; max-width: 60ch; }

#cc-side h3 { font-family: 'Young Serif', Georgia, serif; font-weight: 400; margin: 4px 0 2px; }
#cc-side .cc-hint { font-size: 0.9rem; opacity: 0.75; margin-bottom: 8px; }
.cc-prompt button, button.cc-prompt { justify-content: flex-start !important; text-align: left !important; }

#cc-send { background: var(--cc-olive) !important; color: #fff !important; border: none !important; }
#cc-send:hover { background: #26301F !important; }

#cc-footer { text-align: center; font-size: 0.85rem; opacity: 0.7; padding: 14px 0 4px; border-top: 1px solid rgba(127,127,127,0.25); margin-top: 8px; }
"""


async def handle_chat(user_message, history):
    """Show the user's message and a placeholder, then swap in the agent's answer."""
    history = history or []
    if not user_message or not user_message.strip():
        yield history
        return

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Looking that up…"},
    ]
    yield history

    try:
        response_text = await chat_with_agent(user_message, history[:-2])
    except Exception as e:
        response_text = (
            f"Couldn't reach the agent: {e}\n\n"
            "Check that your Gemini API key (GOOGLE_API_KEY) is set and that server.py "
            "and the data files are in this folder."
        )
    history[-1] = {"role": "assistant", "content": response_text}
    yield history


with gr.Blocks(title="Connoisseur Companion") as demo:
    gr.HTML(
        """
        <div id="cc-header">
          <h1>Connoisseur Companion</h1>
          <p>Your guide to California's restaurant scene. Ask about a place by name,
          search by cuisine or mood, or pull up what reviewers said.</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=520,
                show_label=False,
                placeholder="**Where should we eat?**<br>Ask a question below or pick a suggestion on the right.",
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    show_label=False,
                    placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", elem_id="cc-send", scale=1)

        with gr.Column(scale=1, elem_id="cc-side"):
            gr.HTML("<h3>Try asking</h3><div class='cc-hint'>Tap a question to send it.</div>")
            prompt_buttons = [
                (gr.Button(label, size="sm", elem_classes="cc-prompt"), text)
                for label, text in QUICK_PROMPTS
            ]
            clear_btn = gr.Button("Clear chat", size="sm", variant="secondary")

    gr.HTML(
        "<div id='cc-footer'>Connoisseur Companion · Powered by Google Gemini and the Model Context Protocol</div>"
    )

    # Send on Enter or on the Send button, then clear the textbox
    for trigger in (msg_input.submit, send_btn.click):
        trigger(handle_chat, [msg_input, chatbot], [chatbot])
        trigger(lambda: "", None, msg_input)

    # Quick-prompt buttons send a pre-written question
    for btn, text in prompt_buttons:
        btn.click(handle_chat, [gr.State(text), chatbot], [chatbot])

    clear_btn.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    print("Starting Connoisseur Companion...")
    demo.launch(share=True, theme=gr.themes.Soft(), css=CSS)
