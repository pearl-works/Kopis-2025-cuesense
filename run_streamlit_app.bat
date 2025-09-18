@echo off


REM ---- Create and activate venv (first run only) ----
if not exist ".venv" (
    py -m venv .venv
)
call .venv\Scripts\activate

REM ---- Install requirements ----
py -m pip install --upgrade pip
pip install -r requirements.txt

REM ---- Launch Streamlit ----
-- streamlit run performance_recommendation.py
--streamlit run app.py
streamlit run app_sementic.py