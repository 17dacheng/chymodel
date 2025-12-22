#!/bin/bash

# CHYModel S1131数据集5折交叉验证训练脚本

echo "开始CHYModel S1131数据集5折交叉验证训练..."
echo "日期: $(date)"
echo "========================================"

# 设置路径
DATA_PATH="/home/chengwang/code/chymodel/s1131.csv"
PDB_BASE_PATH="/home/chengwang/data/SKEMPI/PDBs_fixed"
OUTPUT_DIR="/home/chengwang/code/chymodel/s1131_results"

# 默认参数
BATCH_SIZE=4
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
NUM_EPOCHS=100
LEARNING_RATE=1e-3
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --eval_batch_size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --batch_size BATCH_SIZE           默认批大小 (默认: 32)"
            echo "  --train_batch_size TRAIN_BATCH_SIZE  训练批大小 (默认: 32)"
            echo "  --eval_batch_size EVAL_BATCH_SIZE    评估批大小 (默认: 64)"
            echo "  --num_epochs NUM_EPOCHS           训练轮数 (默认: 50)"
            echo "  --learning_rate LEARNING_RATE     学习率 (默认: 1e-3)"
            echo "  --device DEVICE                   设备 (cuda/cpu, 默认: cuda)"
            echo "  --output_dir OUTPUT_DIR           输出目录 (默认: ./s1131_results)"
            echo "  -h, --help                        显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

# 检查PDB目录是否存在
if [ ! -d "$PDB_BASE_PATH" ]; then
    echo "警告: PDB目录不存在: $PDB_BASE_PATH"
    echo "如果使用真实特征，请确保PDB文件存在"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "配置参数:"
echo "  数据路径: $DATA_PATH"
echo "  PDB路径: $PDB_BASE_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  训练批大小: $TRAIN_BATCH_SIZE"
echo "  评估批大小: $EVAL_BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  设备: $DEVICE"
echo "========================================"

# 激活conda环境并运行训练
source /home/chengwang/miniconda3/bin/activate gearbind
python cross_valid.py \
    --data_path "$DATA_PATH" \
    --pdb_base_path "$PDB_BASE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" 

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "训练完成！"
    echo "结果保存在: $OUTPUT_DIR"
    echo ""
    echo "输出文件:"
    ls -la "$OUTPUT_DIR/"
    echo ""
    echo "查看结果摘要:"
    if [ -f "$OUTPUT_DIR/cross_validation_summary.json" ]; then
        python -c "
import json
with open('$OUTPUT_DIR/cross_validation_summary.json', 'r') as f:
    summary = json.load(f)
print(f'平均RMSE: {summary[\"mean_rmse\"]:.4f} ± {summary[\"std_rmse\"]:.4f}')
print(f'平均MAE:  {summary[\"mean_mae\"]:.4f} ± {summary[\"std_mae\"]:.4f}')
print(f'平均R²:   {summary[\"mean_r2\"]:.4f} ± {summary[\"std_r2\"]:.4f}')
print(f'Pearson相关系数: {summary[\"pearson_corr\"]:.4f}')
"
    fi
else
    echo ""
    echo "========================================"
    echo "训练失败！请检查错误信息。"
    exit 1
fi

echo ""
echo "========================================"
echo "脚本执行完成"
echo "日期: $(date)"
echo "========================================"