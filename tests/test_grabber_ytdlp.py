from grabber_ytdlp import GrabberYtDlp


def test_get_username():
    channel_url = "https://www.youtube.com/@kakteper/videos"
    grabber_yt = GrabberYtDlp(channel_url)
    username = grabber_yt.get_channel_name()
    assert username == "Как теперь / проект ОВД-Инфо"

    grabber_yt = GrabberYtDlp(channel_url, offline=True)
    username = grabber_yt.get_channel_name()
    assert username is None


def test_get_channel_id():
    channel_url = "https://www.youtube.com/@kakteper/videos"
    grabber_yt = GrabberYtDlp(channel_url)
    username = grabber_yt.get_channel_id()
    assert username == "@kakteper"

    grabber_yt = GrabberYtDlp(channel_url, offline=True)
    username = grabber_yt.get_channel_id()
    assert username == "@kakteper"


def test_fill_video_info():
    channel_url = "https://www.youtube.com/@kakteper/videos"
    grabber_yt = GrabberYtDlp(channel_url)
    assert grabber_yt.videos

    for video_info in grabber_yt.videos[:5]:
        grabber_yt.fill_video_info(video_info)
        assert video_info.url is not None
        assert not video_info.used_fast_mode
        assert video_info.timestamp is not None
        assert video_info.grabber_tag == grabber_yt.tag
        assert video_info.uploader_id == "@kakteper"
        assert video_info.view_count is not None
        assert video_info.view_count > 0
