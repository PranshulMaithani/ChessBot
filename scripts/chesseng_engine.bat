@echo off
REM Launcher that lichess-bot (or any chess GUI) can point at as an "engine".
REM The architecture flags MUST match how models/chess/best.pt was trained.
REM Current best.pt: 96 channels x 8 residual blocks (the "cloud" preset).
cd /d D:\GoodProjects\Chesseng
python scripts\uci_engine.py --channels 96 --res-blocks 8 %*
