#!/bin/bash
echo "=================================================="
echo "  🚀 [Semantic_Debug] 视觉与 5G/Sionna 联合仿真环境部署..."
echo "=================================================="

# 1. 检查 Mac 多核底层装甲 (供 OpenCV 等 C++ 视觉库并行加速使用)
echo "🔨 正在检查 Mac 底层编译依赖 (libomp)..."
if ! brew ls --versions libomp > /dev/null 2>&1; then
    echo "  -> 未发现 libomp，正在呼叫 Homebrew 进行安装..."
    brew install libomp
else
    echo "  -> libomp 依赖已就绪！"
fi

# 2. 构建防 iCloud 同步的独立虚拟环境
if [ ! -d "venv.nosync" ]; then
    echo "🏗️ 正在构筑虚拟环境隔离舱 (venv.nosync)..."
    python3.12 -m venv venv.nosync
else
    echo "  -> 虚拟环境隔离舱已存在。"
fi

# 3. 装填 Mac 原生通信/视觉弹药
echo "📦 正在安装视觉/通信基础库..."
./venv.nosync/bin/python -m pip install --upgrade pip

# 直接精准读取 Mac 专属依赖清单
./venv.nosync/bin/python -m pip install -r requirements_mac.txt

# 4. 保留 MATLAB 引擎 (在彻底迁移到 Sionna 前作为绝对真理对照组)
echo "🔗 正在安装 MATLAB 官方跨界引擎..."
./venv.nosync/bin/python -m pip install matlabengine==24.2.2

echo "=================================================="
echo "  ✅ 部署彻底竣工！"
echo "  👉 请在终端输入以下命令激活环境："
echo "     source venv.nosync/bin/activate"
echo "=================================================="


# bash "/Users/wtx/Developer/Semantic_Debug/setup_env.sh"
# "/Users/wtx/Developer/Semantic_Debug/venv.nosync/bin/python" -m pip freeze > "/Users/wtx/Developer/Semantic_Debug/requirements_mac.txt"