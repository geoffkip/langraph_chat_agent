from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Define the state of our agent
# This is the "memory" of our agent. It's a dictionary that holds the
# conversation history in a list of messages.
class AgentState(TypedDict):
    """
    Represents the state of our agent.
    The `operator.add` annotation tells LangGraph to append new messages
    to the existing list, rather than replacing it.
    """
    messages: Annotated[list, operator.add]

# 2. Define the Agent's Core Logic
class BasicAgent:
    """
    This class defines the structure and logic of our agent.
    """
    def __init__(self, model):
        self.model = model
        # The graph is the core of the agent, defining the flow of work.
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Builds the computational graph for the agent.
        """
        # Initialize a new state graph with our defined AgentState
        workflow = StateGraph(AgentState)

        # Add a node to the graph. A node is a function or a tool that
        # performs an action. We'll name our node "chatbot".
        workflow.add_node("chatbot", self._chatbot_node)

        # Set the entry point for the graph. When the graph starts, it will
        # call the "chatbot" node first.
        workflow.set_entry_point("chatbot")

        # Set the finish point. In this simple case, the "chatbot" node is
        # also the final step.
        workflow.set_finish_point("chatbot")

        # Compile the graph into a runnable object.
        return workflow.compile()

    def _chatbot_node(self, state: AgentState):
        """
        This is the function that the "chatbot" node will execute.
        It takes the current state (the conversation history) and returns
        the LLM's response.
        """
        # `state["messages"]` contains the conversation history.
        # We invoke the model with this history.
        # The model's response is a new message, which we return.
        return {"messages": [self.model.invoke(state["messages"])]}

# 3. Create the agent instance to be used by the server
#    Instantiate the LLM we want to use (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Create the agent's runnable graph instance
agent_runnable = BasicAgent(llm).graph