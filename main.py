from langchain_core.messages import HumanMessage
from graph import app

def chat():
    # Define a unique thread ID for this specific user session
    config = {"configurable": {"thread_id": "session_001"}}

    print("Conversational RAG Agent (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Ending session. Goodbye!")
            break

        # Package the input as a HumanMessage
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # Invoke the graph
        print("Agent is thinking...")
        result = app.invoke(inputs, config=config)

        # Extract and print the final AI message
        final_message = result["messages"][-1].content
        print(f"AI: {final_message}\n")

if __name__ == "__main__":
    chat()