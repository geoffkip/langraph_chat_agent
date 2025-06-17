import gradio as gr
from fastapi import FastAPI
from app.agent import agent_runnable  # Imports your simple agent

# --- FastAPI Setup ---
# This sets up the underlying server.
app = FastAPI(
    title="LangGraph Agent Server",
    version="1.0",
    description="An API server for a LangGraph agent with a Gradio UI.",
)

# Note: The `add_routes` for the API is removed as it's not needed for the UI-only goal.

# --- Gradio UI with Mermaid Graph Visualization ---

# 1. This function gets the Mermaid diagram text from your agent.
def get_mermaid_graph():
    """Generates the LangGraph graph as a Mermaid diagram string."""
    return agent_runnable.get_graph().draw_mermaid()

# 2. This HTML and JavaScript turns the diagram text into a picture.
#    It loads the Mermaid.js library and tells it to render the graph.
mermaid_html_renderer = f"""
<!DOCTYPE html>
<html>
<head>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true }});
</script>
</head>
<body>
  <pre class="mermaid">
    {get_mermaid_graph()}
  </pre>
</body>
</html>
"""

# 3. This creates the user interface with the two tabs.
with gr.Blocks(theme="soft") as ui:
    gr.Markdown("# My Simple Agent")

    with gr.Tabs():
        # The first tab for chatting (we can add a function later if needed)
        with gr.TabItem("Chat"):
            gr.ChatInterface(fn=lambda message, history: "Chat function not implemented yet.", description="Chat with the agent.")

        # The second tab for showing the graph picture
        with gr.TabItem("View Graph"):
            # The gr.HTML component renders the HTML and JS needed for the picture.
            gr.HTML(value=mermaid_html_renderer)


# --- Mount the Gradio UI onto the FastAPI app ---
# This makes your UI the main page of the server.
app = gr.mount_gradio_app(app, ui, path="/")


# This block lets you run the server with `python app/server.py` if you want.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)