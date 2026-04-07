@echo off
:: 将控制台编码设置为 UTF-8，防止中文字符显示乱码。 
chcp 65001 > nul
:: 启用延迟变量扩展，允许在代码块内部动态修改和读取变量(后续构建命令字符串时用到了) 
setlocal enabledelayedexpansion

:: ========================================
:: Qwen3-ASR 音频转录服务 - 启动脚本 
:: ========================================

:: ====== 【必须配置】请根据实际情况修改 ====== 
:: 模型路径（WSL中的绝对路径）
set MODEL_0_6B=/home/leaf/software/Qwen3-ASR/model/Qwen3-ASR-0.6B
set MODEL_1_7B=/home/leaf/software/Qwen3-ASR/model/Qwen3-ASR-1.7B

:: Python解释器路径 
set PYTHON_PATH=D:\test\.venv\Scripts\python.exe

:: 输出目录配置 
:: 选项:
::   - FIXED: 使用固定目录 (OUTPUT_DIR) 
::   - SAME: 与音频文件同目录 
set OUTPUT_MODE=FIXED
set OUTPUT_DIR=D:\Download\

:: Python客户端脚本路径(与本BAT文件在同一目录时用`%~dp0`) 
set SCRIPT_DIR=C:\Qwen3-ASR_Load\
set PYTHON_SCRIPT=%~dp0asr_client.py

:: WSL配置 
set WSL_DISTRO=Ubuntu-22.04
set WSL_USER=leaf
set PROJECT_PATH=/home/user/Qwen3-ASR

:: ====== 【可选配置】可使用默认值 ======

:: 模型配置 
set GPU_MEMORY=0.7
set SERVICE_PORT=8001
set MAX_MODEL_LEN=1024

:: 是否开启降噪处理(true/false) 
set enable-denoise=true
:: 降噪强度(light/medium/heavy) 
set denoise-strength=medium
:: VAD检测灵敏度(决定人声识别效果：0-1，推荐0.5) 
set vad-threshold=0.5
:: VAD检测静音阈值(秒) 
:: 含义：检测到超过此时长的静音时，会将音频切分为独立片段。此值太小会分太多段，导致识别速度下降明显。如果音频片段太长导致ASR报错，可以适当减小此值 
set vad-min-silence=2
:: 沉默时长大于此值插入换行(秒) 
:: 含义：两个语音片段之间的静音时长超过此值时，在转录文本中插入换行 
set silence-newline=5

:: 单片段最长时长（秒），超出触发二次切分或硬切 
set MAX_CHUNK_DURATION=600
:: 二次VAD切分时使用的静音阈值（秒），比主VAD静音阈值更小以切出更细的片段 
set RETRY_VAD_MIN_SILENCE=0.6

:: 等待服务启动的最大超时时间(秒) 
set WAIT_TIMEOUT=60

:: 是否启用时间戳 (true/false) 此功能未完工
set ENABLE_TIMESTAMP=false

:: ====== 配置区域结束 ======  

:: 设置窗口为黑底绿字 
color 0A
:: 设置命令提示符窗口的标题 
title Qwen3-ASR 音频转录服务 

echo.
echo =============================================== 
echo     Qwen3-ASR 音频转录服务启动器  
echo =============================================== 
echo.

:: 检查Python路径 
if not exist "%PYTHON_PATH%" ( 
    color 0C
    echo [错误] Python解释器不存在: %PYTHON_PATH% 
    echo 请检查配置中的 PYTHON_PATH 路径 
    pause
    exit /b 1
)

:: 检查Python脚本 
if not exist "%PYTHON_SCRIPT%" ( 
    color 0C
    echo [错误] Python客户端脚本不存在: %PYTHON_SCRIPT% 
    echo 请确保 asr_client.py 与本BAT文件在同一目录 
    pause
    exit /b 1
)

:: ====== 界面1：选择模型 ====== 
echo ----------------------------------------------- 
echo  请选择模型：  
echo    [1] Qwen3-ASR-0.6B  (快速，显存占用低)  
echo    [2] Qwen3-ASR-1.7B  (推荐，精度更高)  
echo ----------------------------------------------- 
choice /C 12 /N /M "请输入数字选择模型 [1/2]: "
if errorlevel 2 (
    set MODEL_NAME=%MODEL_1_7B%
    set MODEL_LABEL=Qwen3-ASR-1.7B
) else (
    set MODEL_NAME=%MODEL_0_6B%
    set MODEL_LABEL=Qwen3-ASR-0.6B
)
echo 已选择: !MODEL_LABEL! 
echo.

:: 显示最终配置信息  
echo [配置信息] 
echo   WSL发行版: %WSL_DISTRO%
echo   项目路径: %PROJECT_PATH%
echo   模型: !MODEL_LABEL!
echo   服务端口: %SERVICE_PORT%
echo   时间戳: %ENABLE_TIMESTAMP%
echo   工作模式: (启动后客户端选择) 
echo   输出模式: %OUTPUT_MODE%
if "%OUTPUT_MODE%"=="FIXED" ( 
    echo   输出目录: %OUTPUT_DIR%
)
echo.

echo [降噪和VAD配置] 
echo   降噪: %enable-denoise%
echo   降噪强度: %denoise-strength%
echo   VAD灵敏度: %vad-threshold%
echo   VAD静音阈值: %vad-min-silence% 秒(语音片段切分)  
echo   换行静音阈值: %silence-newline% 秒(文本换行)  
echo   片段最长时长: %MAX_CHUNK_DURATION% 秒 
echo   二次VAD静音阈值: %RETRY_VAD_MIN_SILENCE% 秒 
echo.

:: 构建启动命令
set START_CMD=cd %PROJECT_PATH% ^&^& source .venv/bin/activate ^&^& qwen-asr-serve !MODEL_NAME! --gpu-memory-utilization %GPU_MEMORY% --host 0.0.0.0 --port %SERVICE_PORT% --max-model-len %MAX_MODEL_LEN%

if "%ENABLE_TIMESTAMP%"=="true" (
    set START_CMD=!START_CMD! --enable-timestamp
)

:: 启动WSL服务 
echo [1/3] 正在启动WSL中的Qwen-ASR服务... 
echo.
start "Qwen-ASR-Service" wsl -d %WSL_DISTRO% -u %WSL_USER% bash -c "!START_CMD!"

:: 等待一下让服务窗口打开
timeout /t 2 /nobreak > nul

echo [2/3] 正在启动客户端（客户端将自动等待服务就绪，最多 %WAIT_TIMEOUT% 秒）... 
echo.

:: 传递配置给Python脚本
"%PYTHON_PATH%" "%PYTHON_SCRIPT%" ^
    --port %SERVICE_PORT% ^
    --output-mode %OUTPUT_MODE% ^
    --output-dir %OUTPUT_DIR% ^
    --timeout %WAIT_TIMEOUT% ^
    --enable-denoise %enable-denoise% ^
    --denoise-strength %denoise-strength% ^
    --vad-threshold %vad-threshold% ^
    --vad-min-silence %vad-min-silence% ^
    --silence-newline %silence-newline% ^
    --max-chunk-duration %MAX_CHUNK_DURATION% ^
    --retry-vad-min-silence %RETRY_VAD_MIN_SILENCE%

:: 客户端退出后的清理 
echo.
echo ===============================================
echo 正在关闭WSL服务... 
echo ===============================================

:: 尝试关闭服务
wsl -d %WSL_DISTRO% -u %WSL_USER% bash -c "pkill -f 'qwen-asr-serve'" 2>nul

timeout /t 2 /nobreak > nul
echo 服务已关闭。 
echo.
pause
