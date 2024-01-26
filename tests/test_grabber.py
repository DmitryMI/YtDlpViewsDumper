from grabber import Grabber
from grabber_vk import GrabberVk
from grabber_ytdlp import GrabberYtDlp


def test_get_grabber_for_url():
    yt_url = "https://www.youtube.com/watch?v=Vwv6s_Rfspk"

    grabber_yt_class = Grabber.get_grabber_class_for_url(yt_url)
    assert grabber_yt_class is GrabberYtDlp

    vk_url = "https://vk.com/video-203677279_456240623"
    grabber_vk_class = Grabber.get_grabber_class_for_url(vk_url)
    assert grabber_vk_class is GrabberVk
