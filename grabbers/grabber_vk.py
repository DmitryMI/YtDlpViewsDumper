import json
import logging
import os.path
import random
import re
import time
from urllib.parse import urlparse, parse_qs

import alive_progress
import requests
from alive_progress import alive_bar
from selenium import webdriver
from grabber import Grabber, Credentials, VideoInfo
import webbrowser

VK_API_VERSION = "5.199"


class GrabberVk(Grabber):
    tag = "GrabberVk"
    url_regex = "https?://vk.com/.*"

    def __init__(self, channel_url: str, log_level: str, **kwargs):
        super().__init__(channel_url, log_level)
        self.logger = logging.getLogger("GrabberVk")
        self.logger.setLevel(self._log_level)

        self.offline = kwargs["offline"] if "offline" in kwargs else False

        self.vk_oauth_storage = kwargs["vk_oauth_storage"]
        self.vk_api_key = kwargs["vk_api_key"] if "vk_api_key" in kwargs else None
        if self.vk_api_key is None:
            vk_api_json_path = kwargs["vk_api_json"]
            with open(vk_api_json_path, "r") as file_in:
                vk_api_json = json.load(file_in)
                self.vk_api_key = vk_api_json["id"]

        self.vk_user_id = None
        self.vk_access_token = kwargs["vk_access_token"] if "vk_access_token" in kwargs else None

        if self.vk_access_token is None:
            if os.path.exists(self.vk_oauth_storage):
                with open(self.vk_oauth_storage, "r") as file_in:
                    d = json.load(file_in)
                    self.vk_access_token = d["vk_access_token"]
                    self.vk_user_id = d["vk_user_id"]
                    self.logger.info(f"Using saved VK API access token: {self.vk_access_token}")
            else:
                self.oauth_login()

        if self.vk_access_token is None:
            raise Exception("Not authenticated in VK")

        self._channel_short_name = self._get_group_short_name_from_url()
        self._channel_id = None
        self._channel_name = None
        if not self.offline:
            self._request_group_info()
            self.logger.info(f"VK API responded with group id: {self._channel_id}, group name: {self._channel_name}")
            self._request_video_list()

    def oauth_login(self):
        if self.vk_api_key is None:
            raise Exception("vk_api_key is None!")
        state = random.randint(1000000, 9000000)
        oath_url = f"https://oauth.vk.com/authorize?client_id={self.vk_api_key}&display=page&redirect_uri=https://oauth.vk.com/blank.html&scope=video&response_type=token&v={VK_API_VERSION}&state={state}"
        driver = webdriver.Firefox()
        driver.get(oath_url)
        while "https://oauth.vk.com/blank.html" not in driver.current_url:
            time.sleep(1)
        url = driver.current_url
        driver.close()

        url_param_dict = {}
        url_params_substring = url[url.index("#") + 1:]
        url_params = url_params_substring.split("&")
        for url_param_pair in url_params:
            segments = url_param_pair.split("=")
            key, value = segments[0], segments[1]
            url_param_dict[key] = value

        self.vk_access_token = url_param_dict["access_token"]
        self.vk_user_id = url_param_dict["user_id"]
        callback_state = int(url_param_dict["state"])
        if state != callback_state:
            raise Exception("State mismatch!")

        with open(self.vk_oauth_storage, "w") as file_out:
            json.dump({"vk_access_token": self.vk_access_token, "vk_user_id": self.vk_user_id}, file_out, indent=4)

    def _get_group_short_name_from_url(self):
        if not re.match("https?://vk.com/video/@.*", self._channel_url):
            raise Exception(f"Invalid URL to VK video channel: {self._channel_url}")

        url_parsed = urlparse(self._channel_url)
        short_name = url_parsed.path[url_parsed.path.index("@") + 1:]
        return short_name

    def _request_group_info(self):
        payload = {
            "group_id": self._channel_short_name,
            "access_token": self.vk_access_token,
            "v": VK_API_VERSION
        }
        response = requests.post("https://api.vk.com/method/groups.getById", data=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to get group info with code: {response.status_code}")

        data = json.loads(response.content)
        groups = data["response"]["groups"]
        self._channel_id = groups[0]["id"]
        self._channel_name = groups[0]["name"]

    def _request_video_list_once(self, offset):
        payload = {
            "owner_id": f"-{self._channel_id}",
            "access_token": self.vk_access_token,
            "offset": offset,
            "count": 200,
            "v": VK_API_VERSION
        }
        response = requests.post("https://api.vk.com/method/video.get", data=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to get videos with code: {response.status_code}")

        data = json.loads(response.content)
        if "error" in data:
            error_code = data["error"]["error_code"]
            error_message = data["error"]["error_msg"]
            self.logger.error(f"VK API returned error {error_code}: {error_message}")
            return None, None, error_code

        response_field = data["response"]
        count = response_field["count"]
        items = response_field["items"]
        self.logger.debug(f"VK API responded with 'count' == {count} and {len(items)} items")

        if not items:
            return [], count, 0

        video_infos = []
        for item in items:
            video_id = item["id"]
            video_url = f"https://vk.com/video/@{self._channel_short_name}?z=video-{self._channel_id}_{video_id}"
            video_info = VideoInfo(self.tag, video_url, False)
            video_info.timestamp = item["date"]
            video_info.view_count = item["views"]
            video_info.uploader_id = item["owner_id"]
            video_infos.append(video_info)

        self.logger.debug(f"Fetched {offset} video infos via VK API")
        return video_infos, count, 0

    def _request_video_list(self):
        self.logger.info(f"Requesting video infos via VK API")

        video_infos, count, error_code = self._request_video_list_once(0)
        if video_infos is None or count is None:
            return
        self.videos += video_infos
        with alive_bar(count, title="Downloading metadata", theme="classic", force_tty=True,
                       title_length=0) as bar:
            offset = len(video_infos)
            bar(offset, skipped=True)
            while True:
                video_infos, count, error_code = self._request_video_list_once(offset)

                # Too many requests
                if error_code == 6:
                    time.sleep(5)
                    continue

                if len(video_infos) == 0:
                    break
                self.videos += video_infos
                bar(len(video_infos))
                offset += len(video_infos)

    def get_channel_id(self) -> str:
        return self._channel_short_name

    def get_channel_name(self) -> str | None:
        return self._channel_name

    def get_channel_name_safe(self) -> str:
        if self._channel_name is not None:
            return self._channel_name
        return self._channel_short_name

    def fill_video_info(self, video_info: VideoInfo) -> VideoInfo:
        return video_info
