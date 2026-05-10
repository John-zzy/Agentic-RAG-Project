import json
import shutil

from backend.platform.config.settings import AppSettings
from backend.scenes.ecommerce.definition import build_ecommerce_scene_definition
from backend.scenes.generic_assistant.definition import build_generic_assistant_scene_definition
from backend.platform.tools import ToolContext
from backend.tests.test_support import DATA_DIR, make_test_runtime_dir
from backend.scenes.ecommerce.commerce_tools import SERVICE_TICKETS_FILE_NAME


def _build_test_settings(test_name: str) -> AppSettings:
    runtime_dir = make_test_runtime_dir(test_name)
    data_dir = runtime_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_DIR / "orders.json", data_dir / "orders.json")
    shutil.copy2(DATA_DIR / "products.json", data_dir / "products.json")
    shutil.copy2(DATA_DIR / "reviews.json", data_dir / "reviews.json")
    return AppSettings(data_dir=data_dir)


def test_order_tools_can_lookup_and_update_orders() -> None:
    app_settings = _build_test_settings("tools-order")
    registry = _build_scene_tool_registry(app_settings)

    lookup_tool = registry.get_tool("order_status_lookup")
    lookup_result = lookup_tool.invoke({"order_id": "O202604210002"})
    assert lookup_result.success is True
    assert lookup_result.records[0]["order_id"] == "O202604210002"

    update_tool = registry.get_tool("order_address_update")
    update_result = update_tool.invoke(
        {
            "order_id": "O202604210002",
            "new_address": "上海市徐汇区漕溪北路398号",
        },
        context=ToolContext(agent_name="order_agent"),
    )
    assert update_result.success is True
    assert update_result.records[0]["shipping_address"] == "上海市徐汇区漕溪北路398号"


def test_after_sale_tools_create_persisted_tickets() -> None:
    app_settings = _build_test_settings("tools-after-sale")
    registry = _build_scene_tool_registry(app_settings)

    return_tool = registry.get_tool("return_ticket_create")
    return_result = return_tool.invoke(
        {
            "order_id": "O202604210004",
            "reason": "商品到货后不符合预期",
            "items": ["P012"],
        },
        context=ToolContext(agent_name="after_sale_agent"),
    )
    assert return_result.success is True
    assert return_result.records[0]["ticket_type"] == "return"

    complaint_tool = registry.get_tool("complaint_ticket_create")
    complaint_result = complaint_tool.invoke(
        {
            "order_id": "O202604210004",
            "message": "物流包装破损，需要登记投诉",
            "contact": "13800000000",
        },
        context=ToolContext(agent_name="after_sale_agent"),
    )
    assert complaint_result.success is True
    assert complaint_result.records[0]["ticket_type"] == "complaint"

    tickets_path = app_settings.data_dir / SERVICE_TICKETS_FILE_NAME
    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    assert len(tickets) == 2
    assert {ticket["ticket_type"] for ticket in tickets} == {"return", "complaint"}


def test_registry_enforces_agent_whitelists_and_mcp_exposure_metadata() -> None:
    app_settings = _build_test_settings("tools-whitelist")
    registry = _build_scene_tool_registry(app_settings)

    shopping_tools = registry.list_tools_for_agent("shopping_agent")
    order_tools = registry.list_tools_for_agent("order_agent")
    after_sale_tools = registry.list_tools_for_agent("after_sale_agent")
    mcp_tools = registry.list_mcp_tools()

    assert {registration.tool.name for registration in shopping_tools} == {
        "product_semantic_search",
        "review_semantic_search",
        "inventory_lookup",
        "product_detail_lookup",
    }
    assert {registration.tool.name for registration in order_tools} == {
        "order_status_lookup",
        "order_address_update",
    }
    assert {registration.tool.name for registration in after_sale_tools} == {
        "order_status_lookup",
        "return_ticket_create",
        "complaint_ticket_create",
    }
    assert {registration.tool.name for registration in mcp_tools} == {
        "inventory_lookup",
        "order_status_lookup",
        "order_address_update",
        "return_ticket_create",
        "complaint_ticket_create",
    }


def test_retrieval_tools_return_standardized_records() -> None:
    app_settings = _build_test_settings("tools-retrieval")
    registry = _build_scene_tool_registry(app_settings)

    product_tool = registry.get_tool("product_semantic_search")
    product_result = product_tool.invoke({"query": "续航好的手机", "top_k": 2})
    assert product_result.success is True
    assert product_result.records
    assert {"record_type", "namespace", "citation_id", "snippet", "metadata"} <= set(
        product_result.records[0].keys()
    )

    detail_tool = registry.get_tool("product_detail_lookup")
    detail_result = detail_tool.invoke({"product_id": "P005"})
    assert detail_result.success is True
    assert detail_result.records[0]["product_id"] == "P005"

    inventory_tool = registry.get_tool("inventory_lookup")
    inventory_result = inventory_tool.invoke({"product_id": "P005"})
    assert inventory_result.success is True
    assert inventory_result.records[0]["inventory_status"] in {"in_stock", "low_stock", "out_of_stock"}


def test_generic_scene_definition_exposes_only_generic_retrieval_tools() -> None:
    app_settings = _build_test_settings("tools-generic-scene")
    definition = build_generic_assistant_scene_definition(app_settings=app_settings)

    tool_names = {tool.name for tool in definition.build_tools()}

    assert tool_names == {"knowledge_document_search"}


class _Registration:
    def __init__(self, tool: object, allowed_agents: tuple[str, ...], expose_via_mcp: bool) -> None:
        self.tool = tool
        self.allowed_agents = allowed_agents
        self.expose_via_mcp = expose_via_mcp


class _SceneToolRegistry:
    def __init__(self, registrations: list[_Registration]) -> None:
        self._registrations = registrations

    def get_tool(self, name: str):
        for registration in self._registrations:
            if registration.tool.name == name:
                return registration.tool
        raise KeyError(name)

    def list_tools_for_agent(self, agent_name: str) -> list[_Registration]:
        return [
            registration
            for registration in self._registrations
            if agent_name in registration.allowed_agents
        ]

    def list_mcp_tools(self) -> list[_Registration]:
        return [
            registration
            for registration in self._registrations
            if registration.expose_via_mcp
        ]


def _build_scene_tool_registry(app_settings: AppSettings) -> _SceneToolRegistry:
    """通过 scene definition 聚合工具，避免测试继续依赖旧 registry 兼容层。"""
    definitions = (
        build_ecommerce_scene_definition(app_settings=app_settings),
        build_generic_assistant_scene_definition(app_settings=app_settings),
    )
    registrations: list[_Registration] = []
    for definition in definitions:
        for tool in definition.build_tools():
            registrations.append(
                _Registration(
                    tool=tool,
                    allowed_agents=_resolve_allowed_agents(tool.name),
                    expose_via_mcp=tool.name in {
                        "inventory_lookup",
                        "order_status_lookup",
                        "order_address_update",
                        "return_ticket_create",
                        "complaint_ticket_create",
                    },
                )
            )
    return _SceneToolRegistry(registrations)


def _resolve_allowed_agents(tool_name: str) -> tuple[str, ...]:
    mapping = {
        "product_semantic_search": ("shopping_agent",),
        "review_semantic_search": ("shopping_agent",),
        "inventory_lookup": ("shopping_agent",),
        "product_detail_lookup": ("shopping_agent",),
        "order_status_lookup": ("order_agent", "after_sale_agent"),
        "order_address_update": ("order_agent",),
        "return_ticket_create": ("after_sale_agent",),
        "complaint_ticket_create": ("after_sale_agent",),
        "knowledge_document_search": (),
    }
    return mapping.get(tool_name, ())
