@echo off
git add .
git commit -m "Auto commit: %date:~0,4%-%date:~5,2%-%date:~8,2%"
git push origin main
echo Done!
pause