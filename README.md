# IDS-UAV_CN_FCS

A UAV network intrusion detection project with both scikit-learn ensemble and PyTorch backup models, plus UAVThreatBench LLM threat evaluation.

## 依赖安装

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you want faster fuzzy matching for LLM evaluation, install the optional package:

```bash
pip install python-Levenshtein
```

## 关键目录与文件

- `configs/config.py` - 全局路径和模型配置，包括 `LLM_CONFIG` 和 `LLM_OUTPUT_DIR`
- `models/src/model_sklearn.py` - scikit-learn ensemble model 构建
- `models/src/train_sklearn.py` - sklearn 训练管道
- `models/src/evaluate_sklearn.py` - sklearn 模型评估
- `models/src/model_pytorch.py` - PyTorch CNN-BiLSTM 模型定义
- `models/src/train_pytorch.py` - PyTorch 训练脚本（默认禁用）
- `models/src/evaluate_pytorch.py` - PyTorch 模型评估脚本
- `app/llm_analyzer.py` - UAVThreatBench LLM 响应生成与保存
- `models/src/llm_evaluate.py` - LLM 响应比对评估报告生成

## 使用说明

1. 训练与评估 sklearn 模型：

```bash
python models/src/train_sklearn.py
python models/src/evaluate_sklearn.py
```

2. 运行 PyTorch 评估：

```bash
python models/src/evaluate_pytorch.py
```

3. 生成 LLM 威胁响应：

```bash
python app/llm_analyzer.py
```

4. 对 LLM 响应做离线评估：

```bash
python models/src/llm_evaluate.py
```

## LLM 评估路径

The project now defines the following LLM output paths in `configs/config.py`:

- `LLM_OUTPUT_DIR`
- `LLM_OUTPUT_WITH`
- `LLM_OUTPUT_WITHOUT`
- `LLM_OUTPUT_WITH_RESPONSES`
- `LLM_OUTPUT_WITHOUT_RESPONSES`
- `LLM_EVALUATION_REPORT`

These paths are used by `app/llm_analyzer.py` and `models/src/llm_evaluate.py`.

## 备注

- If the DeepSeek/OpenAI API quota is exhausted, `app/llm_analyzer.py` can fallback to heuristic threat generation when no API key is available.
- The PyTorch training script is kept as a backup option and may be disabled by default if your environment has runtime issues.
