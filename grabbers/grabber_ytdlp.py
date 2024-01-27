import logging
from datetime import datetime
from urllib.parse import urlparse

import yt_dlp

from grabber import Grabber, Credentials, VideoInfo

yt_dlp_logger = logging.getLogger("yt_dlp")
logger = logging.getLogger("GrabberYtDlp")


class GrabberYtDlp(Grabber):
    tag = "GrabberYtDlp"
    url_regex = None

    def __init__(self, channel_url, log_level: str = "INFO", **kwargs):
        super().__init__(channel_url, log_level)

        logger.setLevel(log_level)

        if "fast" in kwargs:
            self._fast = bool(kwargs["fast"])
        else:
            self._fast = False

        if "offline" in kwargs and kwargs["offline"]:
            parsed = urlparse(channel_url)
            path = parsed.path
            path_segments = path.split("/")
            if "youtube" in parsed.hostname:
                self._channel_id = path_segments[1]
                self._username = None
            else:
                raise NotImplemented(f"Offline username parser does not support url {channel_url}")
            return

        self._yt_dlp_params = {
            'quiet': True,
            'no_warnings': True,
            'default_search': 'auto',
            'source_address': '0.0.0.0',
            "extract_flat": True,
            "no-sponsorblock": True,
            "logger": yt_dlp_logger,
        }

        if self._fast:
            logger.warning("Fast extraction enabled. Video metadata will be inaccurate")
            self._yt_dlp_params["extractor_args"] = {'youtubetab': {'approximate_date': "a"}}

        if "credentials" in kwargs and kwargs["credentials"] is not None :
            self._yt_dlp_params["username"] = kwargs["credentials"].username
            self._yt_dlp_params["password"] = kwargs["credentials"].password

        self._ytdl = yt_dlp.YoutubeDL(self._yt_dlp_params)
        self._channel_data = self._ytdl.extract_info(channel_url, download=False)
        if "channel" in self._channel_data:
            self._username = self._channel_data["channel"]
        else:
            logger.error("Cannot determine channel username from channel data!")
            self._username = None

        if "id" in self._channel_data:
            self._channel_id = self._channel_data["uploader_id"]
        else:
            raise Exception("Cannot determine channel id from channel data!")

        for video_data in self._channel_data["entries"]:
            if video_data is None:
                logger.error("Entry is None!")
                continue
            if video_data["_type"] != "url":
                raise Exception(
                    "yt-dlp returned a hierarchy of playlists. If the target service is Youtube, specify the exact tab \
                     (videos, shorts or live) via the URL")

            video_info = VideoInfo(grabber_tag=self.tag, url=video_data["url"], used_fast_mode=self._fast)

            if "upload_date" in video_data:
                upload_date_str = video_data["upload_date"]
                upload_date = datetime.strptime(upload_date_str, '%Y%m%d')
                video_info.timestamp = upload_date.timestamp()

            self.videos.append(video_info)

    def get_channel_name(self):
        return self._username

    def get_channel_id(self):
        return self._channel_id

    def fill_video_info(self, video_info: VideoInfo):

        if self._fast:
            return

        video_data = self._ytdl.extract_info(video_info.url, download=False, process=False)
        video_info.view_count = video_data["view_count"]

        upload_date_str = video_data["upload_date"]
        upload_date = datetime.strptime(upload_date_str, '%Y%m%d')
        video_info.timestamp = upload_date.timestamp()
        video_info.used_fast_mode = False
        video_info.uploader_id = video_data["uploader_id"] if "uploader_id" in video_data else None

        return video_info
