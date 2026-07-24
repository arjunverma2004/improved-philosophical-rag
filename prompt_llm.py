from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

def get_brain(model_choice="gemini-3.1-flash-lite"):
    # Initialize the Gemini model using the API key from your environment
    llm = ChatGoogleGenerativeAI(model=model_choice, temperature=0.7)

    template = """
    You are a careful reader of philosophical texts.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Answer ONLY using the provided context.
    - Explain the philosophical ideas in clear terms.
    - If the text includes reasoning or arguments, break them down step-by-step.
    - Maintain fidelity to the author's meaning (do not distort or oversimplify).
    - Do NOT add external philosophical knowledge or modern interpretations.
    - If unclear or missing, respond:
      "The text does not provide a clear answer."

    Structure your response as:
    1. Direct answer
    2. Explanation (based on the text)
    3. Key idea summary

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Return the LangChain pipeline
    return prompt | llm | StrOutputParser()