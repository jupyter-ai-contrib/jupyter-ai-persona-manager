"""
Pydantic models for MCP server descriptions and the MCP server config file
(`.jupyter/mcp_settings.json`).

These follow the schemas set by Agent Client Protocol.
"""

from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field

class EnvVariable(BaseModel):
    name: Annotated[str, Field(description="The name of the environment variable.")]
    value: Annotated[str, Field(description="The value to set for the environment variable.")]

class McpServerStdio(BaseModel):
    args: Annotated[
        List[str],
        Field(description="Command-line arguments to pass to the MCP server."),
    ]
    command: Annotated[str, Field(description="Path to the MCP server executable.")]
    env: Annotated[
        List[EnvVariable],
        Field(description="Environment variables to set when launching the MCP server."),
    ]
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]

class HttpHeader(BaseModel):
    name: Annotated[str, Field(description="The name of the HTTP header.")]
    value: Annotated[str, Field(description="The value to set for the HTTP header.")]

class McpServerHttp(BaseModel):
    type: Literal['http']
    headers: Annotated[
        List[HttpHeader],
        Field(description="HTTP headers to set when making requests to the MCP server."),
    ]
    name: Annotated[str, Field(description="Human-readable name identifying this MCP server.")]
    url: Annotated[str, Field(description="URL to the MCP server.")]

class McpSettings(BaseModel):
    """
    The Pydantic model defining the schema of the MCP server settings file,
    `.jupyter/mcp_settings.json`.
    """
    mcp_servers: list[Union[McpServerStdio, McpServerHttp]]