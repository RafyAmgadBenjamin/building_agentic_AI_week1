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
from perplexia_ai.tools.calculator import Calculator
from langchain_core.tools import tool 
from langgraph.prebuilt import create_react_agent


class QuestionState(TypedDict):
    question: str
    category: str
    response: str
    history: List[Dict[str, str]]


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
        # Initialize the classifier LLM (no tools needed for classification)
        self.llm_classifier = init_chat_model(model="gpt-4o-mini")

        self.llm_factual = init_chat_model(model="gpt-4o-mini")
        self.llm_analytical = init_chat_model(model="gpt-4o-mini")
        self.llm_comparisons = init_chat_model(model="gpt-4o-mini")
        self.llm_definitions = init_chat_model(model="gpt-4o-mini")
        # Bind the calculator tool to the calculation LLM
        
        llm_calculation = init_chat_model(model="gpt-4o-mini")
        self.calculation_agent = create_react_agent(
            llm_calculation,
            [self._get_calculator_tool()]
        )

        self.graph = StateGraph(QuestionState)
        self.graph.add_node("classify_question", self.__classifier_llm)
        self.graph.add_node("factual_response", self.__response_factual)
        self.graph.add_node("analytical_response", self.__response_analytical)
        self.graph.add_node("comparisons_response", self.__response_comparisons)
        self.graph.add_node("definitions_response", self.__response_definitions)
        self.graph.add_node("calculation_response", self.__response_calculation)

        self.graph.add_edge(START, "classify_question")
        self.graph.add_conditional_edges(
            "classify_question",  # from node
            self.__route_by_category   # function that returns the next node name
        )
        self.graph.add_edge("factual_response", END)
        self.graph.add_edge("analytical_response", END)
        self.graph.add_edge("comparisons_response", END)
        self.graph.add_edge("definitions_response", END)
        self.graph.add_edge("calculation_response", END)
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
        history = self.__process_history_message(chat_history, message)
        # TODO: Students implement query understanding using LangGraph
        result = self.ready_graph.invoke({'question': message, 'history': history})
        return result['response']

    def __process_history_message(self, chat_history: List[Dict[str, str]], message: str) -> List[Dict[str, str]]:
        """Process chat history into the format required by LangChain LLMs."""
        history = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] in ["user", "assistant"]:
                    history.append((msg["role"], msg["content"]))
        # Step 2: Add the new user message
        history.append(("user", message))
        return history

    def __classifier_llm(self, state: QuestionState) -> str:
        """Classify the type of question using an LLM.
        """
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system","""You are expert who can classify the questions into five categories: 'Factual', 'Analytical', 'Comparisons', 'Definitions', 'Calculation'.
            Use 'Calculation' for any mathematical questions, arithmetic operations, percentages, or numerical computations.
            You must respond with only one of these categories."""),
            ("user", "{question}")
        ])

        message = classification_prompt.format_messages(question=state['question'])
        response = self.llm_classifier.invoke(message)
        print("Classified category:", response.content.strip())
        if response.content.strip() not in ['Factual', 'Analytical', 'Comparisons', 'Definitions', 'Calculation']:
            state['category'] = 'Factual'  # Default category
        else:
            state['category'] = response.content.strip()
        return state

    def __response_factual(self, state: QuestionState) -> str:
        """Generate a factual response using its dedicated LLM."""
        print("Generating factual response for question:", state['question'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant who provides direct and concise factual answers."),
            *state['history']  # This unpacks the history as (role, content) tuples
        ])
        messages = prompt.format_messages()
        response = self.llm_factual.invoke(messages)
        state['response'] = response.content.strip()
        return state

    def __response_analytical(self, state: QuestionState) -> str:
        """Generate an analytical response using its dedicated LLM."""
        print("Generating analytical response for question:", state['question'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant who provides in-depth analytical answers including reasoning steps."),
            *state['history']  # This unpacks the history as (role, content) tuples
        ])
        messages = prompt.format_messages()
        response = self.llm_analytical.invoke(messages)
        state['response'] = response.content.strip()
        return state

    def __response_comparisons(self, state: QuestionState) -> str:
        """Generate a comparisons response using its dedicated LLM."""
        print("Generating comparisons response for question:", state['question'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant who provides structured formats(tables, bullet points)."),
            *state['history']  # This unpacks the history as (role, content) tuples
        ])
        messages = prompt.format_messages()
        response = self.llm_comparisons.invoke(messages)
        state['response'] = response.content.strip()
        return state

    def __response_definitions(self, state: QuestionState) -> str:
        """Generate a definitions response using its dedicated LLM."""
        print("Generating definitions response for question:", state['question'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant who provides precise definitions including examples and use cases."),
            *state['history']  # This unpacks the history as (role, content) tuples
        ])
        messages = prompt.format_messages()
        response = self.llm_definitions.invoke(messages)
        state['response'] = response.content.strip()
        return state

    def __response_calculation(self, state: QuestionState) -> str:
        """Generate a calculation response using the agent with calculator tool."""
        print("Generating calculation response for question:", state['question'])
        
        # Invoke the agent - it will automatically execute tools
        result = self.calculation_agent.invoke({
                "messages": state['history']  # Pass the full history here
        })
        
        # Extract the final response from the messages
        state['response'] = result["messages"][-1].content
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
        elif category == "Calculation":
            return "calculation_response"
        else:
            return "factual_response"
    
    def _get_calculator_tool(self):
        """Create and return the calculator tool."""
        @tool
        def calculator_tool(expression: str) -> str:
            """A tool to evaluate any mathematical expression, including arithmetic, percentages, and calculations. Always use this tool for math questions instead of solving them manually."""
            print("Calculating expression tool is invoked:", expression)
            result = Calculator.evaluate_expression(expression)
            return str(result)
        return calculator_tool