@echo off
rem Elevated payload: add the Desktop C++ workload (incl. Windows SDK) to VS 2022.
rem Install config is in the response file; --wait is NOT a valid modify option.
set LOG=C:\Users\User\sdk_modify.log
echo started %DATE% %TIME%> "%LOG%"
start /wait "" "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify --in C:\Users\User\sdk_response.json --quiet --norestart
echo setup exit %ERRORLEVEL%>> "%LOG%"
echo finished %DATE% %TIME%>> "%LOG%"
