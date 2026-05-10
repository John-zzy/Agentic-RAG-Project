from backend.application.runtime.bootstrap import bootstrap_runtime
from backend.platform.config.settings import AppSettings
from backend.tests.test_chat_api import FakeKnowledgeService
from backend.tests.test_support import make_test_runtime_dir


def test_bootstrap_runtime_uses_default_scene_definition() -> None:
    runtime_dir = make_test_runtime_dir("runtime-bootstrap")
    app_settings = AppSettings(
        app={"active_scene": "generic_assistant"},
        session={"sqlite_path": runtime_dir / "bootstrap-sessions.db"},
    )

    chat_service, summary = bootstrap_runtime(
        app_settings,
        knowledge_service=FakeKnowledgeService(),  # type: ignore[arg-type]
    )

    assert chat_service is not None
    assert summary.active_scene == "generic_assistant"
    assert summary.sqlite_path == app_settings.session.sqlite_path
