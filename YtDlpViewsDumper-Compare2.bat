@echo off

set /p url1="First Channel URL: "
set /p url2="Second Channel URL: "

python YtDlpViewsDumper.py --channel "%url1%" "%url2%" --ma_degree 9 --ma_separate --milestone_file "WarInUkraineMilestones.txt" --date_from 01.01.2021

pause