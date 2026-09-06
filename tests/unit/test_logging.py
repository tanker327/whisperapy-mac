import json
import logging

from loguru import logger

from app.config import Settings
from app.core.logging import setup_logging


def test_text_sink_and_stdlib_interception(tmp_path, capsys):
    setup_logging(Settings(debug=True, _env_file=None))
    logging.getLogger("uvicorn.error").warning("from stdlib")
    logger.info("from loguru")
    out = capsys.readouterr().out
    assert "from stdlib" in out
    assert "from loguru" in out
    assert "no-request" in out  # default request id when outside a request
    # Source location should be this test module, not logging/__init__.py.
    assert "test_logging" in out
    assert "callHandlers" not in out


def test_json_format_and_rotating_file(tmp_path, capsys):
    log_file = tmp_path / "logs" / "app.log"
    setup_logging(
        Settings(log_format="json", log_file=log_file, debug=False, _env_file=None)
    )
    with logger.contextualize(request_id="req-1"):
        logger.info("structured line")
    logger.complete()  # flush the enqueue=True file sink

    out_line = next(
        line for line in capsys.readouterr().out.splitlines() if "structured" in line
    )
    record = json.loads(out_line)["record"]
    assert record["message"] == "structured line"
    assert record["extra"]["request_id"] == "req-1"

    assert log_file.exists()
    assert "structured line" in log_file.read_text()


def test_debug_flag_controls_level(capsys):
    setup_logging(Settings(debug=False, _env_file=None))
    logger.debug("hidden")
    logger.info("shown")
    out = capsys.readouterr().out
    assert "hidden" not in out and "shown" in out
