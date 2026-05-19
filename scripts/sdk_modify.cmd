@echo off
rem Elevated payload: add the Desktop C++ workload (incl. Windows SDK) to VS 2022.
rem start /wait blocks for the GUI-subsystem VS bootstrapper; literal quoting via cmd.
set LOG=C:\Users\User\sdk_modify.log
echo started %DATE% %TIME%> "%LOG%"
start /wait "" "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify --installPath "C:\Program Files\Microsoft Visual Studio\2022\Community" --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --quiet --norestart --wait
echo setup exit %ERRORLEVEL%>> "%LOG%"
echo finished %DATE% %TIME%>> "%LOG%"
