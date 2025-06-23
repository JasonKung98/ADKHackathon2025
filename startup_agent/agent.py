from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from toolbox_core import ToolboxSyncClient
# --- OpenAPI Tool Imports ---
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

import os # Required for path operations

GEMINI_MODEL = "gemini-2.0-flash"

# --- 1. Define Main Agent ---
# Agent 1: Data Catalogue API Retrieval Agent
# --- Sample OpenAPI Specification (JSON String) ---
# A basic example of retrieving HTTP response from Malaysia's official OpenDOSM API
openapi_spec_string = """
{
  "openapi": "3.0.0",
  "info": {
    "title": "Foreign Direct Investment Flows API",
    "description": "The Data Catalogue API is designed to allow users to access data from the data catalogue programmatically. It provides a set of endpoints that can be used to query the data based on various parameters and filters.",
    "version": "1.0.0"
  },
  "servers": [
    {
      "url": "https://api.data.gov.my/"
    }
  ],
  "paths": {
    "/data-catalogue": {
      "get": {
        "summary": "Retrieve FDI flows data",
        "operationId": "getFDIFlowsData",
        "parameters": [
          {
            "name": "id",
            "in": "query",
            "description": "Identifier for the data (e.g., fdi_flows)",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "limit",
            "in": "query",
            "description": "Maximum number of results to return",
            "schema": {
              "type": "integer",
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful operation",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "$ref": "#/components/schemas/getFDIFlowsData"
                  }
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "DataCatalogueEntry": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "description": "Unique identifier for the data catalogue entry"
          },
          "title": {
            "type": "string",
            "description": "Title of the data catalogue entry"
          },
          "description": {
            "type": "string",
            "description": "Description of the data catalogue entry"
          }
        }
      },
      "getFDIFlowsData": {
        "type": "object",
        "properties": {
          "net": {
            "type": "number",
            "format": "float"
          },
          "date": {
            "type": "string",
            "format": "date"
          },
          
          "inflow": {
            "type": "number",
            "format": "float"
          },
          "outflow": {
            "type": "number",
            "format": "float"
          }
        }
      }
    }
  }
}
"""
# --- Create OpenAPIToolset ---
openapi_toolset = OpenAPIToolset(
    spec_str=openapi_spec_string,
    spec_str_type='json',
    # No authentication needed for "https://api.data.gov.my/"
)

# --- Agent Definition ---
openapi_agent = LlmAgent(
    name="openapi_agent",
    model=GEMINI_MODEL,
    tools=[openapi_toolset], # Pass the list of RestApiTool objects
    instruction="""You are an information retrieval assistant to obtain FDI flows information via an API.
    Use the available tools to fulfill user requests. You must wait for user input before proceeding the request.
    Do not call any subagents first unless this task is completed. You must also display the response back to the user.
    """,
    description="Retrieve FDI flows using tools generated from an OpenAPI spec.",
    output_key="openapi_result"
)

# --- 2. Define Sub-Agents (to run in sequence) ---

# Sub-Agent 1: Market Research Agent
google_search_agent = LlmAgent(
     name="google_search_agent",
     model=GEMINI_MODEL,
     instruction="""
        Agent Role: Google Search
        Tool Usage: Exclusively use the Google Search tool.

        Overall Goal: To provide the search results back to the user based on their prompts.
        You must wait for user input before proceeding the request,
        otherwise do not proceed anything. You must also display the response back to the user.
    """,
     description="Agent to answer questions using Google Search.",
     # google_search is a pre-built tool which allows the agent to perform Google searches.
     tools=[google_search],
     # Store result in state for the merger agent
     output_key="google_search_result"
)

# Sub-Agent 2: MCP Server for Data-Logging
# It's good practice to define paths dynamically if possible,
# or ensure the user understands the need for an ABSOLUTE path.
# For this example, we'll construct a path relative to this file,
# assuming '/path/to/your/folder' is in the same directory as agent.py.
# REPLACE THIS with an actual absolute path if needed for your setup.
TARGET_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-folder")
# Ensure TARGET_FOLDER_PATH is an absolute path for the MCP server.
# If you created ./adk_agent_samples/mcp_agent/your_folder,

document_agent = LlmAgent(
    model=GEMINI_MODEL,
    name='document_agent',
    instruction=""" Help the user manage their files. You can list files, read files, etc. You must wait for user input before proceeding the request,
    otherwise do not proceed anything. You must also display the response back to the user.
    """,
    description='Retrieve information from specified folder.',
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=[
                    "-y",  # Argument for npx to auto-confirm install
                    "@modelcontextprotocol/server-filesystem",
                    # IMPORTANT: This MUST be an ABSOLUTE path to a folder the
                    # npx process can access.
                    # Replace with a valid absolute path on your system.
                    # For example: "/Users/youruser/accessible_mcp_files"
                    # or use a dynamically constructed absolute path:
                    os.path.abspath(TARGET_FOLDER_PATH),
                ],
            ),
            # Optional: Filter which tools from the MCP server are exposed
            # tool_filter=['list_allowed_directories', 'list_directory', 'read_file']
        )
    ],
    # Store result in state for the merger agent
     output_key="document_result"
)

# Sub-Agent 3: MCP Toolbox for BigQuery
# --- Load Tools from Toolbox ---
# TODO(developer): Ensure the Toolbox server is running at http://127.0.0.1:5000 (default port is 5000)
with ToolboxSyncClient("http://127.0.0.1:5000") as toolbox_client:
    # TODO(developer): Replace "my-toolset" with the actual ID of your toolset as configured in your MCP Toolbox server.
    agent_toolset = toolbox_client.load_toolset("my-toolset")

bigquery_agent = LlmAgent(
     name="bigquery_agent",
     model=GEMINI_MODEL,
     instruction="""
        You're a helpful assistant for managing the BigQuery inventory and order databases.You must wait for user input before proceeding the request,
    otherwise do not proceed anything. If you cannot display the query table as response, just skip it.
    """,
     description="A helpful AI assistant that can manage the BigQuery records in inventory items and orders.",
     tools=agent_toolset, # Pass the loaded toolset
     # Store result in state for the merger agent
     output_key="bigquery_result"
)

# --- 3. Define the Merger Agent (Runs *after* the sequential agents) ---
# This agent takes the results stored in the session state by the sequential agents
# and synthesizes them into a single, structured response with attributions.
merger_agent = LlmAgent(
     name="SynthesisAgent",
     model=GEMINI_MODEL,  # Or potentially a more powerful model if needed for synthesis
     instruction="""You are an AI Assistant responsible for combining results findings into a structured report, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.
 Your entire response MUST be grounded and Do NOT add any external knowledge, facts, or details not present in these specific summaries.**
 """,
     description="Combines output findings from sequential agents into a structured, cited report, strictly grounded on provided inputs.",
     # No tools needed for merging
     # No output_key needed here, as its direct response is the final output of the sequence
 )

 # --- 3. Create the SequentialAgent ---
 # This is the main agent that will be run. It first executes the OpenAPI Agent for FDI Flows, then Google Search Agent
 # for market analysis, then executes the DocumentAgent and merge them to produce the final report, lastly execute the BigQuery Agent for checking inventory list.
sequential_pipeline_agent = SequentialAgent(
     name="ResearchAndSynthesisPipeline",
     # Run OpenAPI, Google Search, DocumentAgent then BigQuery Agent
     sub_agents=[openapi_agent, google_search_agent, document_agent, merger_agent, bigquery_agent],
     description="Coordinates parallel research and synthesizes the results."
 )

root_agent = sequential_pipeline_agent