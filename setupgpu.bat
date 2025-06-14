@echo off
setlocal enabledelayedexpansion

REM === CONFIGURATION ===
set ENV_NAME=tf_gpu_env
set PYTHON=python
set TENSORFLOW_PKG=tensorflow
REM You can set this to tensorflow==2.x.x or tensorflow-gpu if needed

REM === STEP 1: Create virtual environment ===
if not exist "%ENV_NAME%" (
    echo Creating virtual environment: %ENV_NAME%
    %PYTHON% -m venv %ENV_NAME%
) else (
    echo Virtual environment already exists: %ENV_NAME%
)

REM === STEP 2: Activate the environment ===
call "%ENV_NAME%\Scripts\activate.bat"

REM === STEP 3: Upgrade pip and install TensorFlow ===
echo Upgrading pip and installing TensorFlow...
python -m pip install --upgrade pip
python -m pip install %TENSORFLOW_PKG%

REM === STEP 4: Create symbolic links to NVIDIA shared libraries ===
for /f "delims=" %%i in ('python -c "import tensorflow; print(tensorflow.__file__)"') do set tf_path=%%i
for %%i in ("%tf_path%") do set tf_dir=%%~dpi
cd /d "%tf_dir%"

for /D %%D in ("..\nvidia\*\lib") do (
    for %%F in ("%%D\*.so*") do (
        if not exist "%%~nxF" (
            echo Creating symlink: %%~nxF -> %%F
            mklink "%%~nxF" "%%F"
        )
    )
)

cd -

REM === STEP 5: Link ptxas to Scripts folder ===
for /f "delims=" %%i in ('python -c "import nvidia.cuda_nvcc as nvcc; print(nvcc.__file__)"') do set nvcc_path=%%i
for %%i in ("%nvcc_path%") do (
    set nvcc_base=%%~dpi
    for %%j in ("!nvcc_base:~0,-1!") do set nvcc_root=%%~dpj
)

set ptxas_path=
for /r "%nvcc_root%" %%F in (ptxas.exe) do (
    set ptxas_path=%%F
    goto :found_ptxas
)

:found_ptxas
if defined ptxas_path (
    set link_path=%VIRTUAL_ENV%\Scripts\ptxas.exe
    if not exist "%link_path%" (
        echo Creating symlink: %link_path% -> %ptxas_path%
        mklink "%link_path%" "%ptxas_path%"
    )
) else (
    echo WARNING: ptxas.exe not found!
)

REM === STEP 6: Verify GPU Setup ===
echo Verifying TensorFlow GPU access...
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

endlocal
pause
