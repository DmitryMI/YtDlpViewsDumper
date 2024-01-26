from grabber import Grabber, Credentials


class GrabberVk(Grabber):
    url_regex = "https?://vk.com/.*"

    def __init__(self, channel_url: str, offline: bool, creds: Credentials | None):
        super().__init__(channel_url, offline, creds)
