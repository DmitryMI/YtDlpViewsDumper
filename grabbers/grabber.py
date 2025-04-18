from utils import find_subclasses_tags_regex, find_subclasses_with_tags


class VideoInfo:
    def __init__(self, grabber_tag: str, url: str, used_fast_mode: bool):
        self.grabber_tag = grabber_tag
        self.url = url
        self.used_fast_mode = used_fast_mode
        self.view_count = None
        self.timestamp = None
        self.uploader_id = None
        self.title = None

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d: dict):
        video_info = VideoInfo(None, None, False)

        for k, v in d.items():
            if not hasattr(video_info, k):
                continue

            setattr(video_info, k, v)

        return video_info


class Credentials:
    def __init__(self, username, password):
        self.username = username
        self.password = password


class Grabber:
    tag = None
    url_regex = None

    def __init__(self, channel_url: str, log_level: str = "INFO"):
        self._channel_url = channel_url
        self._log_level = log_level
        self.videos: list[VideoInfo] = []

    def fill_video_info(self, video_info: VideoInfo, process_timeout: float | None) -> VideoInfo:
        raise NotImplemented()

    def get_channel_name(self) -> str | None:
        raise NotImplemented()

    def get_channel_name_safe(self) -> str:
        raise NotImplemented()

    def get_channel_id(self) -> str:
        raise NotImplemented()

    @staticmethod
    def get_grabber_class_for_url(url: str):
        clazz_list = list(find_subclasses_tags_regex(Grabber, "grabbers", "url_regex", url))
        if not clazz_list:
            fallback_clazz_list = list(find_subclasses_with_tags(Grabber, "grabbers", "tag", "GrabberYtDlp"))
            return fallback_clazz_list[0]

        if len(clazz_list) > 1:
            raise Exception(f"Multiple grabbers match the url {url}")

        return clazz_list[0]
