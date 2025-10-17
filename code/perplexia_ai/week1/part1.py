"""Part 1 - Query Understanding implementation using LangGraph.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type using conditional edges
- Present information professionally
"""

from typing import Dict, List, Optional, TypedDict

from perplexia_ai.core.chat_interface import ChatInterface

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

class QuestionState(TypedDict):
    question: str
    category: str
    response: str


class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding using LangGraph."""
    
    def __init__(self):
        """Initialize components for query understanding using LangGraph.
        
        Students should:
        - Initialize the chat model
        - Build a graph with classifier and response nodes
        - Set up conditional edges for routing based on query category
        - Compile the graph
        """
        # TODO: Students implement initialization
        self.llm_classifier = init_chat_model(model="gpt-4o-mini")

        self.llm_factual = init_chat_model(model="gpt-4o-mini")
        self.llm_analytical = init_chat_model(model="gpt-4o-mini")
        self.llm_comparisons = init_chat_model(model="gpt-4o-mini")
        self.llm_definitions = init_chat_model(model="gpt-4o-mini")

        self.graph = StateGraph(QuestionState)
        self.graph.add_node("classify_question", self.__classifier_llm)
        self.graph.add_node("factual_response", self.__response_factual)
        self.graph.add_node("analytical_response", self.__response_analytical)
        self.graph.add_node("comparisons_response", self.__response_comparisons)
        self.graph.add_node("definitions_response", self.__response_definitions)
        self.graph.add_edge(START, "classify_question")
        self.graph.add_conditional_edges(
            "classify_question",  # from node
            self.__route_by_category   # function that returns the next node name
        )
        self.graph.add_edge("factual_response", END)
        self.graph.add_edge("analytical_response", END)
        self.graph.add_edge("comparisons_response", END)
        self.graph.add_edge("definitions_response", END)
        self.ready_graph = self.graph.compile()

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding with LangGraph.
        
        Students should:
        - Initialize state with the question
        - Invoke the graph
        - Extract and return the response
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement query understanding using LangGraph
        result = self.ready_graph.invoke({'question': message})
        return result['response']

    def __classifier_llm(self, state: QuestionState) -> str:
        """Classify the type of question using an LLM.
        """
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system","You are expert who can classify the questions into for categories: 'Factual', 'Analytical', 'Comparisons', 'Definitions', you must respond with only one of these categories"),
            ("user", "{question}")
        ])

        message = classification_prompt.format_messages(question=state['question'])
        response = self.llm_classifier.invoke(message)
        print("Classified category:", response.content.strip())
        if response.content.strip() not in ['Factual', 'Analytical', 'Comparisons', 'Definitions']:
            state['category'] = 'Factual'  # Default category
        else:
            state['category'] = response.content.strip()
        return state

    def __response_factual(self, state: QuestionState) -> str:
        """Generate a factual response using its dedicated LLM."""
        print("Generating factual response for question:", state['question'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant who provides direct and concise factual answers."),
            ("user", "Question: {question}\nCategory: Factual\nProvide a well-formatted response based on the category.")
        ])
        messages = prompt.format_messages(question=state['question'])
        response = self.llm_factual.invoke(messages)
        state['response'] = response.content.strip()
        return state

    def __response_analytical(self, state: QuestionState) -> str:
        """Generate an analytical response using its dedicated LLM."""
        print("Generating analytical response for question:", state['question'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant who provides in-depth analytical answers including reasoning steps."),
            ("user", "Question: {question}\nCategory: Analytical\nProvide a well-formatted response based on the category.")
        ])
        messages = prompt.format_messages(question=state['question'])
        response = self.llm_analytical.invoke(messages)
        state['response'] = response.content.strip()
        return state

    def __response_comparisons(self, state: QuestionState) -> str:
        """Generate a comparisons response using its dedicated LLM."""
        print("Generating comparisons response for question:", state['question'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant who provides structured formats(tables, bullet points)."),
            ("user", "Question: {question}\nCategory: Comparisons\nProvide a well-formatted response based on the category.")
        ])
        messages = prompt.format_messages(question=state['question'])
        response = self.llm_comparisons.invoke(messages)
        state['response'] = response.content.strip()
        return state

    def __response_definitions(self, state: QuestionState) -> str:
        """Generate a definitions response using its dedicated LLM."""
        print("Generating definitions response for question:", state['question'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant who provides precise definitions including examples and use cases."),
            ("user", "Question: {question}\nCategory: Definitions\nProvide a well-formatted response based on the category.")
        ])
        messages = prompt.format_messages(question=state['question'])
        response = self.llm_definitions.invoke(messages)
        state['response'] = response.content.strip()
        return state

    @staticmethod
    def __route_by_category(state: QuestionState) -> str:
        category = state.get("category", "Factual")
        if category == "Factual":
            return "factual_response"
        elif category == "Analytical":
            return "analytical_response"
        elif category == "Comparisons":
            return "comparisons_response"
        elif category == "Definitions":
            return "definitions_response"
        else:
            return "factual_response"
    