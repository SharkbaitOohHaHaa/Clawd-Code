@echo off
set "CLAWD_ROOT=%~dp0"
set "HOME=%CLAWD_ROOT%Clawd Codex Home"
set "USERPROFILE=%CLAWD_ROOT%Clawd Codex Home"
powershell -NoProfile -ExecutionPolicy Bypass -File "%CLAWD_ROOT%Clawd Codex Home\.clawd\Review-ClawdUpdate.ps1" %*
