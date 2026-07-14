# configs/config.py
import os
from pathlib import Path

# -------------------- 项目根目录 --------------------
# 自动获取项目根目录（假设config.py在configs/下）
ROOT_DIR = Path(__file__).resolve().parent.parent

# -------------------- 数据路径 --------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # 存放UAV-NIDD原始CSV
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # 存放清洗/标准化后的数据

# UAV-NIDD 数据集文件 (根据你实际下载的文件名调整)
UAV_NIDD_FILES = {
    "uav_case1": RAW_DATA_DIR / "UAV-Case1-Label.csv",  # 攻击从无人机发起
    "ap_case2": RAW_DATA_DIR / "Access-Point-Case2-Label.csv", # 从AP发起
    "gcs_case3": RAW_DATA_DIR / "GCS-Case3-Label.csv",  # 从地面站发起
    "sample": RAW_DATA_DIR / "Sample.csv"
}

# UAVThreatBench 数据集 (JSON格式)
UAV_THREAT_BENCH_DIR = RAW_DATA_DIR / "UAV-ThreatBench"  # 建议将文件放在此子目录下

# 用于评估 (含标准威胁)
UAV_THREAT_BENCH_WITH = UAV_THREAT_BENCH_DIR / "plausible_uav_ot_cyber_scenarios_withThreats.json"

# 用于实际推理 (无威胁)
UAV_THREAT_BENCH_WITHOUT = UAV_THREAT_BENCH_DIR / "plausible_uav_ot_cyber_scenarios_withoutThreats.json"


# -------------------- 模型与训练参数 --------------------
# 模型保存路径
MODEL_DIR = ROOT_DIR / "models" / "outputs" / "checkpoints"
LOG_DIR = ROOT_DIR / "models" / "outputs" / "logs"
PLOT_DIR = ROOT_DIR / "models" / "outputs" / "plots"

# 训练超参数 (可根据后续实验调整)
TRAIN_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "test_size": 0.2,          # 测试集比例
    "val_size": 0.2,           # 验证集占训练集比例
    "random_state": 42,        # 保证可复现
    "use_smote": True,         # 是否使用SMOTE处理不平衡
}

# 模型架构参数 (以CNN-BiLSTM为例)
MODEL_CONFIG = {
    "cnn_out_channels": 64,
    "cnn_kernel_size": 3,
    "lstm_hidden_size": 128,
    "lstm_num_layers": 2,
    "dropout_rate": 0.3,
    "num_classes": 11,         # UAV-NIDD有11种类别 (正常+10攻击)
    "input_dim": 45,           # UAV Case1 特征维度，不同case可能不同
}

# -------------------- 大模型 API 配置 (DeepSeek-V4) --------------------
# 从环境变量读取敏感信息 (需要安装 python-dotenv)
LLM_CONFIG = {
    "api_key": os.getenv("sk-b5af794365e34fe4b8706cba69088004"),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"),
    "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),  # 或 deepseek-reasoner
    "temperature": 0.2,
    "max_tokens": 500,
}

# -------------------- 推理与API服务配置 (Flask) --------------------
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,        # 生产环境设为False
    "use_llm": True,       # 是否启用大模型辅助分析
}

# 确保必要的目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 打印关键路径，方便验证
print(f"项目根目录: {ROOT_DIR}")
print(f"原始数据目录: {RAW_DATA_DIR}")