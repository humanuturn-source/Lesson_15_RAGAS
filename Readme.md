# Install Instructions

## Step 1: Install Ollama & Pull Local Models

Install Ollama to host your models locally, then pull Google DeepMind’s Gemma 4 (or any preferred multimodal vision model, like LLaVA or MiniCPM).

1.	Install Ollama:
⚬	macOS / Linux: Download from ollama.com or run:

    ```bash
    brew install ollama
    ```
⚬	Windows: Download and run the setup installer from ollama.com.

2.	Pull the Local Gemma 4 & 2 Models:
Open your terminal and run:

    ```bash
    ollama pull gemma:2b
    ollama pull gemma4
    ollama pull nomic-embed-text
    ```

## Step 2: Create a Python Virtual Environment

Keep your dependencies isolated by creating a clean virtual environment:

    ```bash
    # 1. Navigate to your project folder
    cd ai_app

    # 2. Create the virtual environment
    python3 -m venv venv

    # 3. Activate the virtual environment
    # macOS / Linux:
    source venv/bin/activate

    # Windows (PowerShell):
    # .\venv\Scripts\Activate.ps1
    ```

## Step 3: Install dependencies

    ```text
    fastapi>=0.100.0
    uvicorn>=0.20.0
    pydantic>=2.0.0
    ollama>=0.1.0
    fastmcp>=0.1.0
    networkx>=3.0
    chromadb>=0.4.0

    ```


    ```bash
    pip install -r requirements.txt
    ```

    ```bash
    pip install ollama pypdf langchain_text_splitters
    pip install "deepeval>=3.0.0" "ragas>=0.2.0" pytest 
    pip install chromadb networkx fastmcp pydantic
    pip install -U langchain-ollama
    pip install pandas matplotlib seaborn

    ```
    

## Step 4: Source documents 

    Copy source documents and Python scripts from the GitHub repositoey

## Step 5: Load Docuemnts

    Run  ingest_graph.py and load all the source documents

## Step 6: Custom Local Judge Class for Ragas

    Copy local_judge.py

## Step 7: Local Pytest Regression Suite (DeepEval Integration)
    Copy run_ragas_evals.py

## Step 8: Run tests via CLI

    ```bash
    python3 run_ragas_evals.py

    ```

## Step 9: Run server wrapper

    Run Server wrapper from the commandline

## Step 10: Run OpenWebUI as a Docker container 

    Verify Open WebUI in Docker Desktop

## Step 11: Connect Open WebUI to Your Local Agent

1.	Open your browser and go to http://localhost:3000.
2.	Create your local admin account upon first login.
3.	Click on Admin Panel (bottom-left avatar) -> Settings -> Connections.
4.	In the OpenAI API section:
    ⚬	Set API Base URL to: http://host.docker.internal:8000/v1
    ⚬	Set API Key to: sk-local (or any placeholder string).
        5.	Click the Refresh/Save icon. You will see supermarket-agent appear in the available model dropdown list.

### Step 12: Give Prompt

    Open the browser and give prompt: http://localhost:3000


### Step 13: Generate RAGAS report

```bash
python3 generate_eval_report.py
```

-------------------