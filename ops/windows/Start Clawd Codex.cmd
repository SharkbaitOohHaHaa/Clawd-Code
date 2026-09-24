@echo off
set "CLAWD_ROOT=%~dp0"
set "HOME=%CLAWD_ROOT%Clawd Codex Home"
set "USERPROFILE=%CLAWD_ROOT%Clawd Codex Home"
set "CLAWD_SKILLS_DIR=%CLAWD_ROOT%Clawd Codex Home\.clawd\skills"
set "CLAWD_SKILL_TRUST_DIR=%CLAWD_ROOT%Clawd Codex Home\.clawd\development-pack"
set "CLAWD_MEMORY_DIR=%CLAWD_ROOT%Clawd Codex Home\.clawd\memory"
set "CLAWD_MEMORY_CONTEXT_CHARS=8000"
if exist "%CLAWD_ROOT%Secrets\.env" (
  for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /r /c:"[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_][ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_]*=" "%CLAWD_ROOT%Secrets\.env"`) do (
    set "%%A=%%~B"
  )
)
cd /d "%CLAWD_ROOT%Clawd Codex Workspace"
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -File "%CLAWD_ROOT%Clawd Codex Home\.clawd\Check-ClawdUpdate.ps1"
"%CLAWD_ROOT%Clawd Codex v0.1.0\.venv\Scripts\clawd.exe" %*
