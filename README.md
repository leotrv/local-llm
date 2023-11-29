# local-llm

## Set up the project
1. Clone the repository using `git clone https://github.com/leotrv/local-llm.git`.
2. Navigate into the repository using `cd local-llm`.
3. Create a virtualenv using `pdm init` and answer the questions accordingly (PDM has to be installed).
4. Install the necessary packages using `pdm install`.
5. If you want to use own PDFs, locate them in the /pdfs folder and run vectordb_build.py. This leads to a newly created vector database.
6. Run the app using `streamlit run main.py`

Note: If you have a GPU, make sure to leverage it using the right llama-cpp-python version. Find relevant information here: https://github.com/abetlen/llama-cpp-python#installation