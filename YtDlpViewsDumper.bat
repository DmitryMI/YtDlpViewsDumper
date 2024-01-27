@echo off

set /p url="Channel URL: "

set python_exe=C:\Users\DmitriyPC\AppData\Local\Programs\Python\Python311\python.exe

SET PYTHONPATH=%PYTHONPATH%;grabbers

%python_exe% YtDlpViewsDumper.py --channel "%url%" --milestone_file "WarInUkraineMilestones.txt" --vk_api_json vk_api.json --offline --spline_length 10

pause