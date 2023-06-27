import streamlit as st
import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt
import requests
import logging
from langchain.llms import OpenAI
from PIL import Image, UnidentifiedImageError
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from transformers import OpenAIGPTTokenizer

logging.basicConfig(level=logging.INFO)


def generate_graph(graph, save_as_png=False, file_name="output.png"):
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    img_url = "https://mermaid.ink/img/" + base64_string

    if save_as_png:
        response = requests.get(img_url)
        with open(file_name, "wb") as f:
            f.write(response.content)
    return img_url


def calculate_openai_api_call_cost(tokens):
    return tokens * 0.00002


template = """
        As a specialized MermaidJS code generator, your purpose is to convert natural language inputs or input with previous code and instructions into precise MermaidJS code for generating error-free flowcharts.
        Here are some important points to consider:
    1. The output generated will exclusively consist of code without any additional text.
    2. The resulting code will not include the word "mermaid" itself.
    3. The code should be presented in a single line, avoiding the use of "\n" for line breaks.

    Here's an example format for the output:
      flowchart TD;A[Have Money] --> B[Go Shopping];B --> C[Think What to Buy];C --> D[iPhone];C --> E[Car];C --> F[Laptop];F --> G[Dell];F --> H[Lenovo];F --> I[Asus];
    
    Current conversation:
    {history}
    Human: {input}
    AI:
"""


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(),
    prompt=PromptTemplate(
        input_variables=["history", "input"],
        template=template,
        template_format="f-string",
        validate_template=True,
    ),
)

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

st.set_page_config(page_title="FlowChart Generator", page_icon=":robot:")
st.header("FlowGenius: Transforming Language into Flowcharts")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "FlowGenius is an innovative online platform that effortlessly converts natural language descriptions into visually appealing flowcharts. With FlowGenius, you can bring your ideas to life by simply describing the steps and relationships in plain English. Our advanced language processing algorithms intelligently interpret your input and generate precise MermaidJS code, which can be directly used to create professional flowcharts.This tool is powered by [Langchain](www.langchain.com and [OpenAI](https://openai.com/))"
    )
with col2:
    st.image(image="imag.jpg", width=500)


st.markdown("## Enter Your Text To Convert")


def get_text():
    input_text = st.text_area(
        label="", placeholder="Your Text.......", key="text_input"
    )
    return input_text


language_input = get_text()

st.markdown("### Your Converted FlowChart")

if language_input:
    # try:
    if not language_input.strip():
        st.warning("Please Enter a valid input")
    else:
        try:
            tokens = tokenizer.tokenize(language_input)
            graph = generate_graph(
                conversation.predict(input=language_input),
                save_as_png=True,
                file_name="flowchart.png",
            )
            if graph is None:
                st.error("Invalid input. Please provide a valid input.")
            else:
                logging.info("Flowchart generated successfully.")
                tokens_consumed = len(tokens)
                api_call_cost = calculate_openai_api_call_cost(tokens_consumed)
                print(
                    f"The cost of the API call is ${api_call_cost} for {tokens_consumed} tokens"
                )
                st.image("flowchart.png", width=500)

        except UnidentifiedImageError:
            logging.error("An error occurred while generating the flowchart image.")
            st.error(
                "An error occurred while generating the flowchart. Please try again."
            )

        except requests.exceptions.RequestException:
            logging.error("An error occurred while retrieving the flowchart image.")
            st.error(
                "An error occurred while generating the flowchart. Please try again."
            )
