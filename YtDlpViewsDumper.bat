chcp 65001

@echo off

set /p url="Channel URL: "

set python_exe=python

SET PYTHONPATH=%PYTHONPATH%;grabbers

%python_exe% YtDlpViewsDumper.py --channel "%url%" --milestone_file "WarInUkraineMilestones.txt" --vk_api_json vk_api.json --ma_degree 10 --title_regex ".*КСТАТИ.*"

pause