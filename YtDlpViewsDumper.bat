@echo off

set /p url="Channel URL: "

python YtDlpViewsDumper.py --channel "%url%" --ma_degree 9 --ma_separate --milestone_file "WarInUkraineMilestones.txt"

pause