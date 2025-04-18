chcp 65001

@echo off

set /p url="Channel URL: "

set python_exe=python

SET PYTHONPATH=%PYTHONPATH%;grabbers

%python_exe% -m pip install -U yt-dlp

%python_exe% YtDlpViewsDumper.py --channel "%url%" --milestone_file "WarInUkraineMilestones.txt" --vk_api_json vk_api.json --ma_degree 100 --timeout 30.0 --jobs 8 --date_from 01.01.2020

pause