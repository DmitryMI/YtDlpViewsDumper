@echo off

set /p url1="First Channel URL: "
set /p url2="Second Channel URL: "

set python_exe=C:\Users\DmitriyPC\AppData\Local\Programs\Python\Python311\python.exe

SET PYTHONPATH=%PYTHONPATH%;grabbers

%python_exe% YtDlpViewsDumper.py --channel "%url1%" "%url2%" --ma_degree 9 --milestone_file "WarInUkraineMilestones.txt" --vk_api_json vk_api.json

pause