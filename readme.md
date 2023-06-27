# FlowGenius: Natural Language to Flowchart Conversion

FlowGenius is an innovative online platform that effortlessly converts natural language descriptions into visually appealing flowcharts. With FlowGenius, you can bring your ideas to life by simply describing the steps and relationships in plain English. Our advanced language processing algorithms intelligently interpret your input and generate precise MermaidJS code, which can be directly used to create professional flowcharts. This tool is powered by Langchain and OpenAI.

## Features

- Converts natural language inputs into MermaidJS code for flowchart generation.
- Handles errors in the MermaidJS code and prompts for corrections.
- Provides a user-friendly interface for input and visualization.

1.  Install the required dependencies: pip install -r requirements.txt

2.  Set up your environment variables : cp .env.example .env

3.  Edit the .env file and add your OpenAI API key.

### Usage

1. Run the Streamlit app: streamlit run app.py

2. Access the FlowGenius web interface at http://localhost:8501 in your web browser.

3. Enter your text description in the input area and press the enter button.

4. The resulting flowchart will be displayed below the input area.

### Troubleshooting

1. Ensure that you have provided a valid OpenAI API key in the .env file.

2. Check your internet connection and make sure you can access the OpenAI API.

3. Verify that all required dependencies are installed by running pip install -r requirements.txt again.
