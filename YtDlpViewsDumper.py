import argparse
import concurrent
import getpass
import logging
import os
import os.path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, CancelledError
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

from cache_manager import CacheManager
from grabber import Grabber, VideoInfo, Credentials

LOG_FORMAT = "%(asctime)s [%(name)-8.8s] [%(funcName)-24.24s] [%(levelname)-5.5s]  %(message)s"

SECONDS_IN_MONTH = 60 * 60 * 24 * 30

YT_DLP_FLAGS = []

milestone_list = []

logger = logging.getLogger("main")
yt_dlp_logger = logging.getLogger("yt-dlp")


def get_darker_color(rgb_hex_str):
    if rgb_hex_str[0] == "#":
        rgb_hex_str = rgb_hex_str[1:]

    r = round(int(rgb_hex_str[0:2], 16) * 0.75)
    g = round(int(rgb_hex_str[2:4], 16) * 0.75)
    b = round(int(rgb_hex_str[4:6], 16) * 0.75)

    return f"#{r:02X}{g:02X}{b:02X}"


def get_cache_creation_time(cache_dir, cache_name):
    if not cache_dir:
        cache_dir = os.getcwd()

    cache_file = os.path.join(cache_dir, f"{cache_name}.cache.json")

    if not os.path.exists(cache_file):
        return None

    return os.path.getctime(cache_file)


def generate_periodical_ticks(timestamp_start, timestamp_end, period_seconds=60 * 60 * 24 * 30, max_ticks=25):
    xticks_values = []
    xticks_labels = []

    ticks_num = (timestamp_end - timestamp_start) / period_seconds
    if ticks_num > max_ticks:
        period_seconds = (timestamp_end - timestamp_start) / max_ticks

    timestamp = timestamp_start
    while timestamp < timestamp_end:
        xticks_values.append(timestamp)

        dt_object = datetime.fromtimestamp(timestamp)
        date_str = dt_object.strftime("%d.%m.%Y")
        xticks_labels.append(date_str)
        timestamp += period_seconds

    if timestamp_end not in xticks_values:
        xticks_values.append(timestamp_end)
        dt_object = datetime.fromtimestamp(timestamp)
        date_str = dt_object.strftime("%d.%m.%Y")
        xticks_labels.append(date_str)

    return xticks_values, xticks_labels


def get_bar_data(data_sorted):
    bar_data = {}

    for entry in data_sorted:
        timestamp = entry["date"]
        views = entry["views"]
        if timestamp in bar_data:
            bar_data[timestamp] += views
        else:
            bar_data[timestamp] = views

    return list(bar_data.keys()), list(bar_data.values())


def weight_linear(n, index_shift):
    if abs(index_shift * 2) > n:
        return 0
    return n - abs(index_shift * 2)


def get_spline_smoothing(timestamps, view_counts, spline_degree, spline_factor, spline_length):
    from scipy.interpolate import make_interp_spline, BSpline

    unique_timestamps, indexes = np.unique(timestamps, return_index=True)
    min_timestamp = unique_timestamps.min()
    max_timestamp = unique_timestamps.max()

    views = []
    for i in indexes:
        views.append(view_counts[i])

    # 300 represents number of points to make between T.min and T.max
    if spline_length is not None:
        x_len = spline_length
    else:
        x_len = int(len(unique_timestamps) * spline_factor)

    x_new = np.linspace(min_timestamp, max_timestamp, x_len)
    logger.info(f"Generating spline with {x_len} x values")
    spl = make_interp_spline(unique_timestamps, views, k=spline_degree)  # type: BSpline
    smooth = spl(x_new)

    '''
    y_values = []
    for i, x in enumerate(x_new):
        y = smooth[i]
        y_values.append(y)
    '''
    return x_new, smooth


def get_moving_mean(video_infos_sorted: list[VideoInfo], n, weight_callable):
    x_values = []
    y_values = []
    for i in range(len(video_infos_sorted)):
        views_accumulator = 0
        items_num = min(i + 1, n)
        divisor = 0
        for j in range(items_num):
            index_shift = j - items_num // 2
            # print(index_shift)
            if i + index_shift >= len(video_infos_sorted):
                break
            video_info = video_infos_sorted[i + index_shift]
            views = video_info.view_count
            if views is None or views == 0:
                continue

            weight = 1
            if weight_callable:
                weight = weight_callable(n, index_shift)

            views_accumulator += views * weight
            divisor += weight

        views_average = views_accumulator / divisor

        x_values.append( video_infos_sorted[i].timestamp)
        y_values.append(views_average)

    return x_values, y_values


def dict_split(data_sorted, x_key="timestamp", y_key="views"):
    x_list = []
    y_list = []
    for entry in data_sorted:
        x = entry[x_key]
        y = entry[y_key]

        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def add_vertical_lines():
    for date, text in milestone_list:
        vline_datetime = datetime.strptime(date, '%d.%m.%Y')
        vline_timestamp = vline_datetime.timestamp()
        plt.axvline(x=vline_timestamp, color='r', label=date, linestyle="--")

        bot, top = plt.ylim()

        plt.text(x=vline_timestamp, y=(top - bot) / 2, s=text, rotation="vertical")


def plot(channel_data_dict: dict[Grabber, list[VideoInfo]], title, date_from_seconds, moving_average_degree,
         spline_degree, spline_factor, spline_length,
         separate_dots=False):

    if spline_factor is not None and spline_length is not None:
        raise Exception("--spline_factor and --spline_length are mutually exclusive")

    xticks_fontsize = 8
    xticks_rotation = 25
    marker_size_total = 4
    marker_size_single = 2

    if separate_dots:
        total_plots = 3
    else:
        total_plots = 2

    plt.figure(figsize=(20, 10))
    plt.suptitle(title)

    timestamp_min = None
    timestamp_max = None

    for grabber, video_info_list in channel_data_dict.items():
        video_infos_sorted = sorted(video_info_list, key=lambda video_info: video_info.timestamp)
        channel_data_dict[grabber] = video_infos_sorted

        timestamp_min_local, timestamp_max_local = video_infos_sorted[0].timestamp, video_infos_sorted[-1].timestamp

        if not timestamp_min or timestamp_min_local < timestamp_min:
            timestamp_min = timestamp_min_local

        if not timestamp_max or timestamp_max_local > timestamp_max:
            timestamp_max = timestamp_max_local

    plt.subplot(total_plots, 1, 1)
    plt.title("Accumulated views")
    plt.xlabel("Date")
    plt.ylabel('Views')

    xticks_values, xticks_labels = generate_periodical_ticks(timestamp_min, timestamp_max)
    plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)

    for grabber, video_infos_sorted in channel_data_dict.items():
        timestamps_values = []
        views_values = []
        views_total_values = []

        views_total_counter = 0

        for video_info in video_infos_sorted:
            timestamp = video_info.timestamp
            views = video_info.view_count
            if views is None or views <= 0:
                continue

            views_total_counter += views

            timestamps_values.append(timestamp)
            views_values.append(views)
            views_total_values.append(views_total_counter)
        channel_name = grabber.get_channel_name()
        if channel_name is None:
            channel_name = grabber.get_channel_id()
        plt.plot(timestamps_values, views_total_values, linestyle='-', marker='o', markersize=marker_size_total,
                 label=channel_name)

    plt.legend()

    bot, top = plt.ylim()
    if bot > 0:
        plt.ylim(bottom=0, top=top)

    if date_from_seconds and date_from_seconds > 0:
        left, _ = plt.xlim()
        if left < date_from_seconds:
            plt.xlim(left=date_from_seconds)

    add_vertical_lines()

    plt.subplot(total_plots, 1, 2)
    title_fragments = []

    if moving_average_degree is not None:
        title_fragments.append(f"moving average N = {moving_average_degree}")

    if spline_factor is not None:
        title_fragments.append(f"s-factor = {spline_factor}")
    if spline_degree is not None:
        title_fragments.append(f"s-degree = {spline_degree}")
    if spline_length is not None:
        title_fragments.append(f"s-length = {spline_length}")

    plt.title("Views per video (" + ", ".join(title_fragments) + ")")

    plt.xlabel("Date")
    plt.ylabel('Views')
    plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)

    line_colors = []

    for grabber, video_infos_sorted in channel_data_dict.items():
        if moving_average_degree is not None:
            ma_x, ma_y = get_moving_mean(video_infos_sorted, moving_average_degree, weight_linear)
        else:
            ma_x = [video_info.timestamp for video_info in video_infos_sorted]
            ma_y = [video_info.view_count for video_info in video_infos_sorted]

        if spline_degree is not None and (spline_factor is not None or spline_length is not None):
            ma_x, ma_y = get_spline_smoothing(ma_x, ma_y, spline_degree, spline_factor, spline_length)

        channel_name = grabber.get_channel_name_safe()

        line = plt.plot(ma_x, ma_y, linestyle='-',
                        label=channel_name + " (moving average)")
        if line:
            line_colors.append(line[0]._color)

    plt.legend()

    bot, top = plt.ylim()
    if bot > 0:
        plt.ylim(bottom=0, top=top)
    if date_from_seconds and date_from_seconds > 0:
        left, _ = plt.xlim()
        if left < date_from_seconds:
            plt.xlim(left=date_from_seconds)

    add_vertical_lines()

    if separate_dots:
        plt.subplot(total_plots, 1, 3)
        plt.title("Views per video")
        plt.xlabel("Date")
        plt.ylabel('Views')
        plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)

    for i, (grabber, video_infos_sorted) in enumerate(channel_data_dict.items()):
        timestamps_values = []
        views_values = []

        for video_info in video_infos_sorted:
            views = video_info.view_count
            if views is None or views <= 0:
                continue

            timestamps_values.append(video_info.timestamp)
            views_values.append(views)

        if not separate_dots:

            color = get_darker_color(line_colors[i])

            plt.plot(timestamps_values, views_values, "^", markersize=marker_size_single,
                     label=grabber.get_channel_name_safe(), color=color)
        else:
            plt.plot(timestamps_values, views_values, "o", markersize=marker_size_single,
                     label=grabber.get_channel_name_safe())

    if separate_dots:
        plt.legend()
        if date_from_seconds and date_from_seconds > 0:
            left, _ = plt.xlim()
            if left < date_from_seconds:
                plt.xlim(left=date_from_seconds)
        add_vertical_lines()

    plt.tight_layout(h_pad=1)
    plt.show(block=True)


def find_by_url(video_infos: list[VideoInfo], url):
    if video_infos is None:
        return None
    for video_info in video_infos:
        if url == video_info.url:
            return video_info
    return None


def fetch_channel_data_offline(channel_url, cache_manager, args):
    logger.info("Reading from cache (offline mode)...")

    grabber_class = Grabber.get_grabber_class_for_url(channel_url)
    grabber = grabber_class(channel_url, True, **vars(args))

    username = grabber.get_channel_id()

    dataset = cache_manager.read_cache(username)
    logger.info(f"Read {len(dataset)} videos from cache {username}")

    logger.info(f"Done for {channel_url}!")
    video_infos = []
    for entry in dataset:
        video_info = VideoInfo.from_dict(entry)
        video_infos.append(video_info)

    return video_infos, grabber


def remove_outdated_entries(cached_dataset, cache_manager, timestamp_current, username, cache_expiration_seconds):
    cache_outdated_entries = []
    for cached_entry in cached_dataset:
        cache_timestamp = cached_entry["cache_timestamp"] if "cache_timestamp" in cached_entry else None

        if cache_timestamp is None:
            cache_timestamp = cache_manager.get_cache_creation_time(username)

        if cache_expiration_seconds and \
                0 < cache_expiration_seconds < timestamp_current - cache_timestamp:
            cache_outdated_entries.append(cached_entry)
    logger.info(f"{len(cache_outdated_entries)} entries in cache {username} are outdated")

    for entry in cache_outdated_entries:
        cached_dataset.remove(entry)


def fetch_channel_data(
        channel_url,
        cache_dir,
        jobs,
        date_from_seconds,
        cache_expiration_seconds,
        cache_manager,
        offline,
        fast,
        credentials,
        args
):
    grabber_class = Grabber.get_grabber_class_for_url(channel_url)
    grabber = grabber_class(channel_url, logger.level, **vars(args), credentials=credentials)

    video_infos = []

    timestamp_current = datetime.now().timestamp()

    username = grabber.get_channel_id()

    if offline:
        if cache_manager.cache_exists(username):
            return fetch_channel_data_offline(channel_url, cache_manager, args)
        else:
            logger.error("Cache {} does not exist, impossible to work in offline mode!")
            return None, username

    outdated_cache_num = 0

    logger.info(f"Downloading video list from {channel_url}...")
    if not grabber.videos:
        return None, username

    videos_num = len(grabber.videos)
    logger.info(f"{videos_num} videos found!")

    logger.info("Reading from cache and downloading...")

    cached_dataset = cache_manager.read_cache(username)
    if cached_dataset:
        logger.info(f"Read {len(cached_dataset)} videos from cache {username}")
    else:
        logger.info(f"Cache {username} does not exist or is empty.")

    if fast:
        logger.warning(f"Fast mode enabled, metadata will not be saved to cache")

    remove_outdated_entries(cached_dataset, cache_manager, timestamp_current, username, cache_expiration_seconds)
    logger.info(f"Rewriting cache {username} with outdated elements removed...")
    cache_manager.clear_cache(username)
    cache_manager.append(username, cached_dataset, len(cached_dataset))
    cache_manager.flush_buffers()

    for cache_entry in cached_dataset:
        video_info = VideoInfo.from_dict(cache_entry)
        video_infos.append(video_info)

    video_infos_not_filled = []
    futures = []

    # Filter videos that already have all required information
    for video_info in grabber.videos:
        if video_info.timestamp is not None and video_info.view_count is not None and not video_info.used_fast_mode:
            cached_entry = find_by_url(video_infos, video_info.url)
            if cached_entry is None:
                cached_entry = video_info.to_dict()
                cached_entry["cache_timestamp"] = datetime.now().timestamp()
                cache_manager.append(username, cached_entry)
            video_infos.append(video_info)
        else:
            video_infos_not_filled.append(video_info)

    with (
        ThreadPoolExecutor(max_workers=jobs) as executor,
        # ProcessPoolExecutor(max_workers=jobs) as executor,
        alive_bar(len(video_infos_not_filled), title="Downloading metadata", theme="classic", force_tty=True,
                  title_length=0) as bar
    ):

        for video_info in video_infos_not_filled:
            cached_entry = find_by_url(video_infos, video_info.url)
            if cached_entry is None:
                future = executor.submit(grabber.fill_video_info, video_info)
                futures.append(future)
            else:
                bar(1, skipped=True)

        date_from_reached = False

        for future in concurrent.futures.as_completed(futures, timeout=None):
            try:
                video_info: VideoInfo = future.result()

                video_infos.append(video_info)
                # write_cache(cache_dir, username, video_infos)
                if not fast:
                    cache_entry = video_info.to_dict()
                    cache_entry["cache_timestamp"] = datetime.now().timestamp()
                    cache_manager.append(username, cache_entry)

                upload_date = datetime.fromtimestamp(video_info.timestamp)
                bar.title(upload_date.strftime("%d.%m.%Y"))
                bar(1)

                if date_from_seconds and video_info.timestamp < date_from_seconds:
                    if not date_from_reached:
                        logger.info("Minimum timestamp reached, aborting remaining downloads...")
                    future_index = futures.index(future)

                    if len(futures) > future_index:
                        cancel_futures = list(futures[future_index + 1:])
                        marked_num = 0
                        cancelled_num = 0
                        for cancel_future in cancel_futures:
                            if cancel_future.cancelled():
                                continue
                            marked_num += 1
                            cancelled = cancel_future.cancel()
                            if cancelled:
                                cancelled_num += 1
                        logger.debug(f"Marked {marked_num} futures for cancelling, {cancelled_num} actually cancelled")
                    else:
                        logger.debug(f"No futures will be cancelled")
                    date_from_reached = True

            except CancelledError as err:
                # bar(1, skipped = True)
                pass
            except Exception as err:
                logger.error(f"Downloading error: {err}")

    logger.info(f"Done for {channel_url}!")
    if not fast:
        cache_manager.flush_buffers()
    return video_infos, grabber


def load_vertical_lines(args):
    milestone_lines = []
    if args.milestone:
        for milestone in args.milestone:
            milestone_lines.append(milestone)

    if args.milestone_file:
        if not os.path.isfile(args.milestone_file):
            logger.error(f"File {args.milestone_file} does not exist or is not a file!")
            quit(-1)

        with open(args.milestone_file) as fin:
            milestone_lines += fin.readlines()

    for milestone_line in milestone_lines:
        first_space_index = milestone_line.index(" ")
        date_str = milestone_line[:first_space_index]
        date = datetime.strptime(date_str, '%d.%m.%Y')
        text = milestone_line[first_space_index + 1:]
        text = text.strip()
        logger.info(f"Loaded vertical line: {date.strftime('%d.%m.%Y')} - {text}")
        milestone_list.append((date_str, text))


def main():
    parser = argparse.ArgumentParser(
        prog="Youtube Views dumper",
        description="Collects information about video views using yt-dlp",
        epilog="Delete it immediately!"
    )

    parser.add_argument("-v", "--verbosity", required=False, type=str, default="INFO")
    parser.add_argument("-f", "--fast", required=False, action="store_true", default=False)
    parser.add_argument("--yt_dlp_verbosity", required=False, type=str, default=None)
    parser.add_argument("--yt_dlp", required=False, type=str, default="yt-dlp")
    parser.add_argument("--cache_expiration", required=False, default=30)
    parser.add_argument("--offline", required=False, default=False, action="store_true")
    parser.add_argument("--username", required=False, type=str, default=None)
    parser.add_argument("--password", required=False, type=str, default=None)
    parser.add_argument("--date_from", required=False, type=str, default=None)
    parser.add_argument("--spline", required=False, action="store_true")
    parser.add_argument("--spline_degree", required=False, type=int, default=3)
    parser.add_argument("--spline_factor", required=False, type=float, default=None)
    parser.add_argument("--spline_length", required=False, type=int, default=None)
    parser.add_argument("--ma_degree", required=False, type=int, default=1)
    parser.add_argument("--separate_dots", required=False, action="store_true")
    parser.add_argument("--cache_dir", required=False, type=str, default="cache")
    parser.add_argument("-j", "--jobs", required=False, type=int, default=32)
    parser.add_argument("--channels", type=str, nargs="+")
    parser.add_argument("--milestone", type=str, nargs="+")
    parser.add_argument("--milestone_file", required=False, type=str)
    parser.add_argument("--vk_oauth_storage", required=False, type=str, default="vk_oauth_storage.json")
    parser.add_argument("--vk_access_token", required=False, type=str, default=None)
    parser.add_argument("--vk_api_key", required=False, type=str, default=None)
    parser.add_argument("--vk_api_json", required=False, type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(format=LOG_FORMAT)
    logger.setLevel(args.verbosity)

    '''
    if args.ma_degree is not None and args.spline:
        logger.error("--ma_agree and --spline are mutually exclusive")
        return
    '''
    if args.yt_dlp_verbosity is None:
        yt_dlp_logger.setLevel(args.verbosity)
    else:
        yt_dlp_logger.setLevel(args.yt_dlp_verbosity)

    load_vertical_lines(args)

    cache_dir = args.cache_dir
    cache_expiration_seconds = args.cache_expiration * 24 * 60 * 60
    cache_manager = CacheManager(cache_dir)

    date_from_str = args.date_from
    if date_from_str:
        date_from = datetime.strptime(date_from_str, '%d.%m.%Y')
        date_from_seconds = date_from.timestamp()
        logger.info(f"Minimum timestamp set to {date_from_str}({date_from_seconds} seconds)")
    else:
        date_from = None
        date_from_seconds = None

    moving_average_degree = args.ma_degree

    credentials = None
    if args.username is not None:
        if args.password is not None:
            credentials = Credentials(args.username, args.password)
        else:
            password = getpass.getpass(f"Password for {args.username}: ")
            credentials = Credentials(args.username, password)

    channel_data_dict = {}

    usernames = []

    for channel in args.channels:
        video_infos, grabber = fetch_channel_data(
            channel,
            cache_dir,
            args.jobs,
            date_from_seconds,
            cache_expiration_seconds,
            cache_manager,
            args.offline,
            args.fast,
            credentials,
            args
        )
        if not video_infos:
            continue
        channel_data_dict[grabber] = video_infos
        channel_name = grabber.get_channel_name()
        if channel_name is None:
            channel_name = grabber.get_channel_id()
        usernames.append(channel_name)

    if not channel_data_dict:
        return 0

    plot_title = "Views of " + ", ".join(usernames)
    if date_from_seconds:
        plot_title = f"{plot_title} (from {date_from_str})"

    plot(channel_data_dict, plot_title, date_from_seconds, moving_average_degree, args.spline_degree, args.spline_factor, args.spline_length, args.separate_dots)

    return 0


if __name__ == "__main__":
    main()
