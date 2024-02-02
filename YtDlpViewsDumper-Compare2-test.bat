@echo off

set url1="https://vk.com/video/@temalebedev"
set url2="https://vk.com/video/@ikakprosto"

set python_exe=C:\Users\DmitriyPC\AppData\Local\Programs\Python\Python311\python.exe

SET PYTHONPATH=%PYTHONPATH%;grabbers

%python_exe% YtDlpViewsDumper.py --channel "%url1%" "%url2%" --ma_degree 9 --milestone_file "WarInUkraineMilestones.txt" --vk_api_json vk_api.json --spline_factor 100 --separate_dots

pause