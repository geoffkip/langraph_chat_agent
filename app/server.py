import gradio as gr
from fastapi import FastAPI
from langserve import add_routes
from app.agent import agent_runnable
from langchain_core.messages import HumanMessage, AIMessage

# --- FastAPI and LangServe Setup ---
app = FastAPI(
    title="LangGraph Agent Server",
    version="1.0",
    description="An API server for a LangGraph agent with a Gradio UI.",
)

add_routes(
    app,
    agent_runnable,
    path="/chat",
)

# --- Gradio UI with Mermaid Graph Visualization ---

# 1. Define the chat function (same as before)
def chat_function(message, history):
    formatted_history = []
    for user_msg, ai_msg in history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    
    formatted_history.append(HumanMessage(content=message))
    agent_input = {"messages": formatted_history}
    response = agent_runnable.invoke(agent_input)
    ai_response = response['messages'][-1].content
    return ai_response

# 2. Define a function to generate the Mermaid diagram string
def get_mermaid_graph():
    """Generates the LangGraph graph as a Mermaid diagram string."""
    graph = agent_runnable.get_graph()
    mermaid_string = graph.draw_mermaid()
    return mermaid_string

# 3. Create the HTML/JS needed to render the Mermaid diagram
#    We will inject this into the Gradio UI.
mermaid_html = f"""
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

# 4. Create the Gradio UI with a Tabbed Interface
with gr.Blocks(theme="soft") as ui:
    gr.Markdown("# My LangGraph Agent")
    
    with gr.Tabs():
        with gr.TabItem("Chat"):
            gr.ChatInterface(
                fn=chat_function,
                description="Ask the agent anything!",
            )
        with gr.TabItem("View Graph"):
            # Use an HTML component to render the Mermaid diagram
            gr.HTML(value=mermaid_html)


# --- Mount the Gradio UI onto the FastAPI app ---
app = gr.mount_gradio_app(app, ui, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)