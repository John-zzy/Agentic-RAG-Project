import json
import shutil

from backend.config.settings import AppSettings
from backend.tests.test_support import DATA_DIR, make_test_runtime_dir
from backend.tools.base import ToolContext
from backend.tools.ecommerce.commerce import SERVICE_TICKETS_FILE_NAME
from backend.tools.ecommerce.registry import build_default_tool_registry


def _build_test_settings(test_name: str) -> AppSettings:
    runtime_dir = make_test_runtime_dir(test_name)
    data_dir = runtime_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_DIR / "orders.json", data_dir / "orders.json")
    return AppSettings(data_dir=data_dir)


def test_order_tools_can_lookup_and_update_orders() -> None:
    app_settings = _build_test_settings("tools-order")
    registry = build_default_tool_registry(app_settings)

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
    assert update_result.metadata["updated_by_agent"] == "order_agent"


def test_after_sale_tools_create_persisted_tickets() -> None:
    app_settings = _build_test_settings("tools-after-sale")
    registry = build_default_tool_registry(app_settings)

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
    registry = build_default_tool_registry(app_settings)

    shopping_tools = registry.list_tools_for_agent("shopping_agent")
    order_tools = registry.list_tools_for_agent("order_agent")
    after_sale_tools = registry.list_tools_for_agent("after_sale_agent")
    mcp_tools = registry.list_mcp_tools()

    assert shopping_tools == []
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
        "order_status_lookup",
        "order_address_update",
        "return_ticket_create",
        "complaint_ticket_create",
    }
