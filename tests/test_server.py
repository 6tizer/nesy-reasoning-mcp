import json

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest

from nesy_reasoning_mcp.server import create_server, initialization_options
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import ASSERT_RELATIONS


@pytest.mark.asyncio
async def test_tools_list_returns_eleven_tools_with_schemas() -> None:
    server = create_server(RelationStore())
    handler = server.request_handlers[ListToolsRequest]
    result = await handler(ListToolsRequest(method="tools/list"))
    tools = result.root.tools

    assert [tool.name for tool in tools] == [
        "nesy.assert_relations",
        "nesy.list_relations",
        "nesy.clear_relations",
        "nesy.classify",
        "nesy.verify_chain",
        "nesy.assert_exclusive",
        "nesy.check_contradictions",
        "nesy.load_relations",
        "nesy.export_relations",
        "nesy.summarize_graph",
        "nesy.counterfactual",
    ]
    assert all(tool.inputSchema for tool in tools)
    assert all(tool.outputSchema for tool in tools)


@pytest.mark.asyncio
async def test_server_call_tool_result_shape() -> None:
    server = create_server(RelationStore())
    await server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
    handler = server.request_handlers[CallToolRequest]
    result = await handler(
        CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name=ASSERT_RELATIONS,
                arguments={
                    "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
                    "check_contradictions": False,
                },
            ),
        )
    )

    payload = result.root
    assert payload.isError is False
    assert payload.structuredContent is not None
    assert json.loads(payload.content[0].text) == payload.structuredContent


def test_initialization_options_exposes_tools_capability() -> None:
    server = create_server(RelationStore())
    options = initialization_options(server)

    assert options.server_name == "nesy-reasoning"
    assert options.server_version == "0.6.0"
    assert options.capabilities.tools is not None
    assert options.capabilities.tools.listChanged is False
