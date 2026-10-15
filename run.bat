@echo off
title EduPredict AI - Student Performance Prediction System
echo =======================================================
echo   Starting EduPredict AI ML Prediction Server...
echo =======================================================
python -m pip install -r requirements.txt
echo.
echo Launching Application at http://localhost:5000 ...
python app.py
pause
