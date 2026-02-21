"""
Pydantic models for MCP server descriptions and the MCP server config file
(`.jupyter/mcp_servers.json`).

These follow the schemas set by Agent Client Protocol.
"""

from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field

class EnvVariable(BaseModel):
    # The name of the environment variable.
    name: Annotated[str, Field(description="The name of the environment variable.")]
    # The value to set for the environment variable.
    value: Annotated[str, Field(description="The value to set for the environment variable.")]

class McpServerStdio(BaseModel):
    # Command-line arguments to pass to the MCP server.
    args: Annotated[
        List[str],
        Field(description="Command-line arguments to pass to the MCP server."),
    ]
    # Path to the MCP server executable.
    command: Annotated[str, Field(description="Path to the MCP server executable.")]
    # Environment variables to set when launching the MCP server.
    env: Annotated[
        List[EnvVariable],
        Field(description="Environment variables to set when launching the MCP server."),
    ]
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]

class HttpHeader(BaseModel):
    # The name of the HTTP header.
    name: Annotated[str, Field(description="The name of the HTTP header.")]
    # The value to set for the HTTP header.
    value: Annotated[str, Field(description="The value to set for the HTTP header.")]

class McpServerHttp(BaseModel):
    type: Literal['http']
    # HTTP headers to set when making requests to the MCP server.
    headers: Annotated[
        List[HttpHeader],
        Field(description="HTTP headers to set when making requests to the MCP server."),
    ]
    # Human-readable name identifying this MCP server.
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    # URL to the MCP server.
    url: Annotated[str, Field(description="URL to the MCP server.")]

class McpSettings(BaseModel):
    """
    The Pydantic model defining the schema of the MCP server settings file,
    `.jupyter/mcp_servers.json`.
    """
    mcp_servers: list[Union[McpServerStdio, McpServerHttp]]