# configs/config.py
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import torch
except Exception:  # pragma: no cover
    # 捕获所有异常（包括 Windows 上可能出现的 OSError/DLL 加载错误），
    # 并在无法加载 torch 时回退为 None，使得非 PyTorch 代码仍可导入。
    torch = None

# -------------------- 项目根目录 --------------------
# 自动获取项目根目录（假设config.py在configs/下）
ROOT_DIR = Path(__file__).resolve().parent.parent

# 从项目根目录或配置目录加载 .env，确保 os.getenv() 能拿到 API 配置
if load_dotenv is not None:
    for env_file in (ROOT_DIR / ".env", ROOT_DIR / "configs" / ".env", Path.cwd() / ".env"):
        if env_file.exists():
            load_dotenv(env_file, override=False)
            break

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
# 用于评估 (含标准威胁)
UAV_THREAT_BENCH_WITH = RAW_DATA_DIR / "plausible_uav_ot_cyber_scenarios_withThreats.json"

# 用于实际推理 (无威胁)
UAV_THREAT_BENCH_WITHOUT = RAW_DATA_DIR / "plausible_uav_ot_cyber_scenarios_withoutThreats.json"


# -------------------- 模型与训练参数 --------------------
# 模型保存路径
MODEL_DIR = ROOT_DIR / "models" / "outputs" / "checkpoints"
LOG_DIR = ROOT_DIR / "models" / "outputs" / "logs"
PLOT_DIR = ROOT_DIR / "models" / "outputs" / "plots"

# -------------------- 大模型评估输出路径 --------------------
LLM_OUTPUT_DIR = ROOT_DIR / "models" / "outputs" / "llm_evaluation"
LLM_OUTPUT_WITH = LLM_OUTPUT_DIR / "withThreats"
LLM_OUTPUT_WITHOUT = LLM_OUTPUT_DIR / "withoutThreats"
LLM_OUTPUT_WITH_RESPONSES = LLM_OUTPUT_WITH / "responses.json"
LLM_OUTPUT_WITHOUT_RESPONSES = LLM_OUTPUT_WITHOUT / "responses.json"
LLM_EVALUATION_REPORT = LLM_OUTPUT_WITH / "evaluation_report.txt"

# -------------------- 大模型 API 配置（确认完整性） --------------------
LLM_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "temperature": 0.2,
    "max_tokens": 500,
}

# 确保 LLM 输出目录存在
os.makedirs(LLM_OUTPUT_WITH, exist_ok=True)
os.makedirs(LLM_OUTPUT_WITHOUT, exist_ok=True)

# 训练超参数 (可根据后续实验调整)
TRAIN_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "test_size": 0.2,          # 测试集比例
    "val_size": 0.2,           # 验证集占训练集比例
    "random_state": 42,        # 保证可复现
    "use_smote": True,         # 是否使用SMOTE处理不平衡
    "early_stopping_patience": 5,
    "device": "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
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

# -------------------- PyTorch 模型配置（备用） --------------------
MODEL_CONFIG_PYTORCH = {
    "cnn_out_channels": 64,
    "lstm_hidden_size": 128,
    "lstm_num_layers": 2,
    "dropout_rate": 0.3,
    "num_classes": 14,          # UAV-NIDD 有 14 个类别（正常 + 13 种攻击）
}

# -------------------- 训练超参数（备用） --------------------
TRAIN_CONFIG_PYTORCH = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "early_stopping_patience": 5,
    "device": "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
}

# -------------------- scikit-learn ensemble 配置 --------------------
SKLEARN_MODEL_CONFIG = {
    "ensemble_type": "VotingClassifier",
    "use_standardizer": False,
}

SKLEARN_RF_CONFIG = {
    "n_estimators": 150,
    "max_depth": 16,
    "random_state": 42,
    "n_jobs": -1,
}

SKLEARN_XGB_CONFIG = {
    "n_estimators": 150,
    "learning_rate": 0.08,
    "max_depth": 8,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "verbosity": 0,
}

SKLEARN_DT_CONFIG = {
    "max_depth": 12,
    "min_samples_split": 8,
    "random_state": 42,
}

SKLEARN_VOTING_CONFIG = {
    "voting": "soft",
    "weights": None,
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
""" print(f"项目根目录: {ROOT_DIR}")
print(f"原始数据目录: {RAW_DATA_DIR}")
print(f"处理后数据目录: {PROCESSED_DATA_DIR}")
print(f"模型目录: {MODEL_DIR}")
print(f"日志目录: {LOG_DIR}")
print(f"图表目录: {PLOT_DIR}")
print(f"LLM 数据 (含威胁): {UAV_THREAT_BENCH_WITH}")
print(f"LLM 数据 (无威胁): {UAV_THREAT_BENCH_WITHOUT}")
print(f"大模型 API 配置: {LLM_CONFIG}") """