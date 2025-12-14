@echo off
REM --- 1. Replace the path below with your actual Anaconda activation script path ---
call C:\Users\User\anaconda3\Scripts\activate.bat base

REM --- 2. Run the Streamlit app ---
streamlit run app.py

REM --- 3. Pause the window so you can see any output/errors ---
pause