# Batch级别训练损失输出功能

## 概述

修改了五折交叉验证训练框架，现在在每个batch训练和验证后都会打印详细的损失信息，提供更细粒度的训练监控。

## 修改内容

### 1. 训练循环 (`train_epoch`方法)

**之前:**
```python
for batch in train_loader:
    # ... 训练代码 ...
    total_loss += loss.item()
```

**现在:**
```python
for batch_idx, batch in enumerate(train_loader):
    # ... 训练代码 ...
    batch_loss = loss.item()
    total_loss += batch_loss
    print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {batch_loss:.6f} | Predicted: {ddg_pred.item():.4f} | True: {ddg_true.item():.4f}")
```

### 2. 验证循环 (`validate`方法)

**之前:**
```python
with torch.no_grad():
    for batch in val_loader:
        # ... 验证代码 ...
        total_loss += loss.item()
```

**现在:**
```python
with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        # ... 验证代码 ...
        batch_loss = loss.item()
        total_loss += batch_loss
        print(f"  Val Batch {batch_idx+1}/{len(val_loader)} | Loss: {batch_loss:.6f} | Predicted: {ddg_pred.item():.4f} | True: {ddg_true.item():.4f}")
```

### 3. Epoch级别的输出格式

**改进后的层次化输出:**

```
Epoch 1/50
==================================================
Training:
  Batch 1/8 | Loss: 2.456789 | Predicted: 1.2345 | True: 1.5678
  Batch 2/8 | Loss: 2.123456 | Predicted: 0.9876 | True: 1.2345
  ...
  Batch 8/8 | Loss: 1.987654 | Predicted: 1.1111 | True: 0.8888

Validation:
  Val Batch 1/4 | Loss: 2.111111 | Predicted: 1.2222 | True: 1.3333
  Val Batch 2/4 | Loss: 1.999999 | Predicted: 0.7777 | True: 1.0000
  ...
  Val Batch 4/4 | Loss: 1.888888 | Predicted: 1.4444 | True: 1.2222

Epoch 1 Summary:
  Time: 15.2s | Avg Train Loss: 2.1234 | Avg Val Loss: 1.9999 | Val RMSE: 1.4142 | Val R²: 0.2500 | Val Pearson: 0.5000
```

## 输出信息说明

### Batch级别信息
- **Batch序号**: 当前batch在总batch数中的位置
- **Loss**: 当前batch的MSE损失值
- **Predicted**: 模型对当前batch样本的ΔΔG预测值
- **True**: 当前batch样本的真实ΔΔG值

### Epoch级别总结
- **Time**: 当前epoch耗时
- **Avg Train Loss**: 训练集平均损失
- **Avg Val Loss**: 验证集平均损失
- **Val RMSE**: 验证集均方根误差
- **Val R²**: 验证集决定系数
- **Val Pearson**: 验证集皮尔逊相关系数

## 使用示例

### 1. 测试脚本
运行测试脚本验证输出格式:
```bash
python test_training_output.py
```

### 2. 演示脚本
运行完整的交叉验证演示:
```bash
python demo_cross_valid.py
```

### 3. 实际训练
运行真实的五折交叉验证:
```bash
python cross_valid.py \
    --data_path /path/to/skemi_data.csv \
    --pdb_base_path /path/to/pdbs \
    --batch_size 8 \
    --num_epochs 50 \
    --device cuda
```

## 优势

1. **细粒度监控**: 可以看到每个batch的训练进度
2. **早期发现异常**: 及时发现异常高的损失值
3. **模型理解**: 观察预测值与真实值的对比
4. **调试便利**: 更容易定位训练过程中的问题
5. **训练可视化**: 便于制作训练过程的日志和图表

## 注意事项

1. **输出量**: 增加了控制台输出量，适合开发调试
2. **性能影响**: 打印操作对训练性能影响极小
3. **日志管理**: 建议将输出重定向到文件以便长期保存
4. **内存使用**: 不影响内存使用，只是增加了I/O操作

## 自定义配置

可以通过修改`print_every`参数控制epoch级别总结的频率:
```python
config = {
    'print_every': 5,  # 每5个epoch打印一次总结
    # ... 其他参数
}
```

如果希望减少batch级别的输出，可以注释掉相应的print语句，或者添加配置选项来控制。