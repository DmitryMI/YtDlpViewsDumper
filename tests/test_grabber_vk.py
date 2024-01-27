from grabber import Credentials
from grabber_vk import GrabberVk


def test_get_channel_id(fix_vk_api_key):
    assert fix_vk_api_key is not None
    vk_oauth_storage = "vk_oauth_storage.json"
    grabber = GrabberVk("https://vk.com/video/@vkvideo", offline=False, vk_oauth_storage=vk_oauth_storage,
                        vk_api_key=fix_vk_api_key, log_level="INFO")

    assert grabber.get_channel_id() == "vkvideo"
    assert grabber.get_channel_name() == "VK Видео"

    assert len(grabber.videos) >= 339

