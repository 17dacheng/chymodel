# ComplexDDG S1131数据集训练指南

本项目提供了在S1131数据集上进行ComplexDDG模型5折交叉验证的完整代码。

## 文件结构

```
complexddg/
├── complexddg_model.py    # ComplexDDG模型定义
├── feature_extractor.py    # 特征提取模块
├── dataset.py             # 数据处理模块（新增）
├── cross_valid.py         # 交叉验证训练脚本（已修改）
├── train_s1131.sh         # S1131数据集训练脚本
├── s1131.csv             # S1131数据集
└── README_S1131.md       # 本文件
```

## 主要修改

### 1. 创建 `dataset.py`
- 包含 `SKEMPIDataset` 类
- 包含数据加载和划分相关函数
- 支持真实特征提取和虚拟特征测试

### 2. 修改 `cross_valid.py`
- 使用新的 `dataset.py` 模块
- 默认训练50个epochs
- 支持分别设置训练和评估的batch size
- 在s1131数据集上进行5折交叉验证

## 使用方法

### 快速开始

```bash
# 使用默认参数训练
./train_s1131.sh
```

### 自定义参数训练

```bash
# 查看所有可用参数
./train_s1131.sh --help

# 自定义参数示例
./train_s1131.sh \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --num_epochs 100 \
    --learning_rate 5e-4 \
    --device cuda \
    --output_dir ./my_results
```

### 测试运行（使用虚拟特征）

```bash
# 使用虚拟特征进行快速测试
./train_s1131.sh --use_dummy_features
```

### 直接使用Python脚本

```bash
python cross_valid.py \
    --data_path /home/chengwang/code/complexddg/s1131.csv \
    --pdb_base_path /home/chengwang/data/SKEMPI/PDBs_fixed \
    --output_dir ./s1131_results \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --num_epochs 50 \
    --learning_rate 1e-3 \
    --device cuda
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | 必需 | S1131数据集CSV文件路径 |
| `--pdb_base_path` | 必需 | PDB文件基础路径 |
| `--output_dir` | `./s1131_results` | 输出目录 |
| `--batch_size` | 32 | 默认批大小 |
| `--train_batch_size` | 32 | 训练批大小 |
| `--eval_batch_size` | 64 | 评估批大小 |
| `--num_epochs` | 50 | 训练轮数 |
| `--learning_rate` | 1e-3 | 学习率 |
| `--device` | `cuda` | 计算设备 (cuda/cpu) |
| `--use_dummy_features` | False | 使用虚拟特征（仅用于测试） |

## 输出结果

训练完成后，输出目录将包含：

```
s1131_results/
├── cross_validation_summary.json    # 交叉验证结果摘要
├── fold_1_best_model.pth            # 第1折最佳模型
├── fold_1_predictions.csv           # 第1折预测结果
├── fold_2_best_model.pth            # 第2折最佳模型
├── fold_2_predictions.csv           # 第2折预测结果
├── ...                               # 其他折的结果
└── fold_5_best_model.pth            # 第5折最佳模型
```

### 结果摘要格式

`cross_validation_summary.json` 包含：
- 各折详细结果
- 平均性能指标（RMSE, MAE, R²）
- 标准差
- 训练配置信息

## 数据集统计

S1131数据集的基本信息：
- 总样本数：8个突变
- 唯一复合物数：8个
- ΔΔG范围：约 -4.2 到 11.2 kcal/mol

## 注意事项

1. **GPU内存**：根据GPU内存大小调整batch size
   - 8GB GPU: 建议 train_batch_size=16, eval_batch_size=32
   - 16GB GPU: 建议 train_batch_size=32, eval_batch_size=64
   - 24GB+ GPU: 可以使用更大的batch size

2. **训练时间**：使用真实特征时，首次训练需要提取特征，可能较长时间

3. **特征缓存**：特征提取结果会自动缓存，后续训练会更快

4. **虚拟特征**：`--use_dummy_features` 仅用于代码测试，结果没有实际意义

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减小batch size
   ./train_s1131.sh --train_batch_size 8 --eval_batch_size 16
   ```

2. **CUDA内存错误**
   ```bash
   # 使用CPU训练
   ./train_s1131.sh --device cpu
   ```

3. **PDB文件不存在**
   - 确保 `--pdb_base_path` 指向正确的PDB文件目录
   - 或者使用 `--use_dummy_features` 进行测试

4. **特征提取失败**
   - 检查PDB文件格式是否正确
   - 确保Biopython等依赖已正确安装

### 依赖安装

```bash
pip install torch pandas numpy scikit-learn biopython transformers esm
```

## 性能优化建议

1. **数据预处理**：可以预先提取所有特征，避免重复计算
2. **批处理**：根据GPU内存适当增大batch size
3. **学习率调度**：可以尝试不同的学习率和调度策略
4. **模型架构**：可以调整hidden_dim、num_layers等超参数

## 引用

如果您使用本代码，请引用相关论文和ESM模型。