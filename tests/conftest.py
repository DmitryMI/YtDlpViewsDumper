import pytest


def pytest_addoption(parser):
    parser.addoption("--vk_api_key", action="store", default=None)


@pytest.fixture(scope="session")
def fix_vk_api_key(pytestconfig):
    return pytestconfig.getoption("vk_api_key")
