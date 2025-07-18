# LLM Zoomcamp - Agents Homework

This project contains the solution for the homework from the "0a-agents" workshop of the LLM Zoomcamp.

The primary goal is to learn and practically apply the concept of **Function Calling** using the **Model-Context Protocol (MCP)**. As part of this project, a simple weather server was created using the `fastmcp` library, along with a client to interact with it.

## Project Structure

-   `weather_server.py`: Implements the MCP server that exposes two tools: `get_weather` and `set_weather`.
-   `mcp_client_test.py`: A client to programmatically interact with the server. It connects to `weather_server.py` and requests the list of available tools.
-   `venv/`: Directory for the Python virtual environment to isolate project dependencies.

## Setup and Installation

### 1. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python3 -m venv venv

# Activate on Linux / macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 2. Install Dependencies

All required libraries should be listed in a `requirements.txt` file.

```bash
pip install -r requirements.txt
```

> **Note:** If you don't have a `requirements.txt` file, you can create one after installing the necessary packages as described below.

### 3. (Optional) Create `requirements.txt`

After installing the required libraries manually, you can freeze their versions into a file for easy replication.

```bash
# First, install the main library
pip install fastmcp

# Then, create the requirements file
pip freeze > requirements.txt
```

## Usage (Homework Walkthrough)

### Question 4: Running the MCP Server

To start the server, run the following command:

```bash
python weather_server.py
```

The output will display information about the server. The answer to Q4 is the transport type shown in the startup logs, for example:
`INFO Starting MCP server 'Demo 🚀' with transport 'stdio'`

### Question 5: Manual Server Interaction

While the server is running, you can send it JSON-RPC requests directly in the same terminal. Copy and paste each line below one by one, pressing Enter after each.

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client"}}}
{"jsonrpc": "2.0", "method": "notifications/initialized"}
{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_weather", "arguments": {"city": "Berlin"}}}
```
The answer to Q5 is the final JSON response that the server returns for the last request.

### Question 6: Using the Python Client

Stop the server (`Ctrl+C`) if it is still running. Then, execute the client script:

```bash
python mcp_client_test.py
```

The script will automatically start the server, connect to it, request the list of available tools, and print the list to the console. This output is the answer to Q6.
