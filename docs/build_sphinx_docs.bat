@echo off
setlocal

echo ********************************************************************************
echo                         Starting the Sphinx script
echo ********************************************************************************
echo                 Creating plugin API files and HTML pages...

rem Resolve all paths relative to the directory containing this batch file.
set "SCRIPT_DIR=%~dp0"

rem Remove old generated API files and build output to avoid obsolete files.
if exist "%SCRIPT_DIR%source\developers\generated" (
    rmdir /s /q "%SCRIPT_DIR%source\developers\generated"
    if errorlevel 1 goto :cleanup_failed
)

if exist "%SCRIPT_DIR%build" (
    rmdir /s /q "%SCRIPT_DIR%build"
    if errorlevel 1 goto :cleanup_failed
)

rem -a writes all output files.
rem -E rebuilds the Sphinx environment without using the saved cache.
rem -b html selects the HTML builder.
sphinx-build -a -E -b html "%SCRIPT_DIR%source" "%SCRIPT_DIR%build"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ERROR: Sphinx build failed with exit code %EXIT_CODE%.
) else (
    echo.
    echo Sphinx documentation built successfully.
    echo Output: "%SCRIPT_DIR%build\index.html"
)

exit /b %EXIT_CODE%

:cleanup_failed
echo.
echo ERROR: Unable to remove an existing generated or build directory.
exit /b 1
