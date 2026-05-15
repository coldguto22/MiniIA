@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Nao foi possivel encontrar o ambiente virtual em .venv.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Ambiente virtual ativado.
echo.
echo Arquivos principais para execucao manual:
echo   - loop_dante.py: iniciar o loop
echo   - alimentar_memoria.py: inputar memoria manualmente
echo   - conversar.py: iniciar o chat direto
echo.
echo Use o nome do arquivo desejado para executar manualmente.
cmd /k