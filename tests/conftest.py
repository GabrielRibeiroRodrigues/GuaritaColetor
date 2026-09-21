import pytest

from fake_api import FakeApi


@pytest.fixture
def api():
    server = FakeApi()
    server.start()
    yield server
    server.stop()
