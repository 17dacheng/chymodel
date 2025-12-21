"""
CHYModel训练和评估脚本
五折交叉验证训练框架
"""

import os
import time
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import argparse

# 导入模型定义和数据集
from model import CHYModelWithGeometric
from dataset import create_dataloader, print_data_statistics


class Trainer:
    """训练器类，封装训练和评估逻辑"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 20
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch in train_loader:
            # 将数据移到设备
            esm_emb = batch['esm_embeddings'].to(self.device)
            foldx_feat = batch['foldx_features'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            ddg_true = batch['ddg'].to(self.device)
            
            # 获取几何图数据（如果存在）
            wt_graph = batch.get('wt_graph', None)
            mt_graph = batch.get('mt_graph', None)
            
            # 前向传播
            self.optimizer.zero_grad()
            if wt_graph is not None and mt_graph is not None:
                # 使用几何特征
                wt_graph = wt_graph.to(self.device)
                mt_graph = mt_graph.to(self.device)
                ddg_pred = self.model(esm_emb, foldx_feat, wt_graph, mt_graph, attention_mask)
            else:
                # 不使用几何特征（向后兼容）
                ddg_pred = self.model(esm_emb, foldx_feat, attention_mask)
            
            loss = self.criterion(ddg_pred, ddg_true)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录指标
            total_loss += loss.item()
            all_predictions.extend(ddg_pred.detach().cpu().numpy())
            all_targets.extend(ddg_true.detach().cpu().numpy())
        
        # 计算指标
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)
        pearson_corr, _ = pearsonr(all_targets, all_predictions)
        
        return {
            'loss': total_loss / len(train_loader),
            'rmse': rmse,
            'r2': r2,
            'pearson_corr': pearson_corr if not np.isnan(pearson_corr) else 0.0
        }
    
    def validate(self, val_loader: DataLoader) -> Tuple[Dict[str, float], List, List]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移到设备
                esm_emb = batch['esm_embeddings'].to(self.device)
                foldx_feat = batch['foldx_features'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                ddg_true = batch['ddg'].to(self.device)
                
                # 获取几何图数据（如果存在）
                wt_graph = batch.get('wt_graph', None)
                mt_graph = batch.get('mt_graph', None)
                
                # 前向传播
                if wt_graph is not None and mt_graph is not None:
                    # 使用几何特征
                    wt_graph = wt_graph.to(self.device)
                    mt_graph = mt_graph.to(self.device)
                    ddg_pred = self.model(esm_emb, foldx_feat, wt_graph, mt_graph, attention_mask)
                else:
                    # 不使用几何特征（向后兼容）
                    ddg_pred = self.model(esm_emb, foldx_feat, attention_mask)
                
                loss = self.criterion(ddg_pred, ddg_true)
                
                # 记录指标
                total_loss += loss.item()
                all_predictions.extend(ddg_pred.cpu().numpy())
                all_targets.extend(ddg_true.cpu().numpy())
        
        # 计算指标
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)
        pearson_corr, _ = pearsonr(all_targets, all_predictions)
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'rmse': rmse,
            'r2': r2,
            'pearson_corr': pearson_corr if not np.isnan(pearson_corr) else 0.0
        }
        
        return metrics, all_predictions, all_targets
    
    def early_stopping(self, val_loss: float) -> bool:
        """早停检查"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.patience


def train_fold(
    fold: int,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """训练单个折"""
    print(f"\n{'='*60}")
    print(f"训练第 {fold+1} 折")
    print(f"{'='*60}")
    
    # 创建临时数据文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='_train.csv', delete=False) as f_train:
        train_data.to_csv(f_train.name, sep='\t', index=False)
        train_path = f_train.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_val.csv', delete=False) as f_val:
        val_data.to_csv(f_val.name, sep='\t', index=False)
        val_path = f_val.name
    
    try:
        # 创建数据加载器
        train_loader = create_dataloader(
            data_path=train_path,
            pdb_base_path=config['pdb_base_path'],
            batch_size=config.get('train_batch_size', config['batch_size']),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            use_dummy_features=config.get('use_dummy_features', False),
            use_geometric_features=config.get('use_geometric_features', True)
        )
        
        val_loader = create_dataloader(
            data_path=val_path,
            pdb_base_path=config['pdb_base_path'],
            batch_size=config.get('eval_batch_size', config['batch_size']),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            use_dummy_features=config.get('use_dummy_features', False),
            use_geometric_features=config.get('use_geometric_features', True)
        )
        
        # 初始化模型和训练器
        device = torch.device(config['device'])
        
        # 根据配置选择模型
        model = CHYModelWithGeometric(
            esm_dim=config.get('esm_embedding_dim', 1280),
            foldx_dim=config.get('foldx_features', 22),
            hidden_dim=config.get('hidden_dims', [512, 256, 128])[0],
            num_heads=config.get('num_attention_heads', 8),
            num_layers=config.get('num_attention_layers', 2),
            dropout=config.get('dropout_rate', 0.1)
        )

        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5),
            patience=config.get('patience', 20)
        )
        
        # 训练循环
        best_model_state = None
        best_metrics = None
        history = {'train': [], 'val': []}
        
        for epoch in range(config['num_epochs']):
            start_time = time.time()
            
            # 训练
            train_metrics = trainer.train_epoch(train_loader)
            
            # 验证
            val_metrics, val_pred, val_true = trainer.validate(val_loader)
            
            # 学习率调度
            trainer.scheduler.step(val_metrics['loss'])
            
            # 记录历史
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)
            
            # 早停检查
            stop_early = trainer.early_stopping(val_metrics['loss'])
            
            # 保存最佳模型
            if val_metrics['loss'] < trainer.best_val_loss:
                best_model_state = model.state_dict().copy()
                best_metrics = val_metrics.copy()
                best_metrics['predictions'] = val_pred
                best_metrics['targets'] = val_true
            
            # 打印进度
            epoch_time = time.time() - start_time
            if (epoch + 1) % config.get('print_every', 10) == 0:
                print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val RMSE: {val_metrics['rmse']:.4f} | "
                      f"Val R²: {val_metrics['r2']:.4f} | "
                      f"Val Pearson: {val_metrics['pearson_corr']:.4f}")
            
            if stop_early:
                print(f"早停在 epoch {epoch+1}")
                break
        
        # 加载最佳模型进行最终评估
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            final_metrics, final_pred, final_true = trainer.validate(val_loader)
        else:
            final_metrics, final_pred, final_true = val_metrics, val_pred, val_true
        
        # 保存模型
        model_save_path = os.path.join(output_dir, f'fold_{fold+1}_best_model.pth')
        torch.save({
            'fold': fold + 1,
            'model_state_dict': best_model_state or model.state_dict(),
            'config': config,
            'metrics': best_metrics or final_metrics,
            'history': history
        }, model_save_path)
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'prediction': final_pred,
            'target': final_true
        })
        results_path = os.path.join(output_dir, f'fold_{fold+1}_predictions.csv')
        results_df.to_csv(results_path, index=False)
        
        return {
            'fold': fold + 1,
            'best_metrics': best_metrics,
            'final_metrics': final_metrics,
            'model_path': model_save_path,
            'results_path': results_path,
            'train_size': len(train_data),
            'val_size': len(val_data)
        }
        
    finally:
        # 清理临时文件
        os.unlink(train_path)
        os.unlink(val_path)


def five_fold_cross_validation(
    data_path: str,
    pdb_base_path: str,
    output_dir: str = './results',
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    执行五折交叉验证
    
    Args:
        data_path: 数据文件路径
        pdb_base_path: PDB文件基础路径
        output_dir: 输出目录
        config: 训练配置
    
    Returns:
        交叉验证结果
    """
    # 默认配置
    default_config = {
        'batch_size': 8,
        'train_batch_size': 8,
        'eval_batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'patience': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_attention_layers': 2,
        'num_attention_heads': 8,
        'dropout_rate': 0.1,
        'print_every': 10,
        'use_dummy_features': False,  # 实际使用时设为False
        'use_geometric_features': True,  # 使用几何特征
        'random_seed': 42
    }
    
    if config:
        default_config.update(config)
    config = default_config
    
    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据并打印统计信息
    print("加载数据...")
    print_data_statistics(data_path)
    data = pd.read_csv(data_path, sep='\t')
    print(f"总数据量: {len(data)} 个突变")
    
    # 按复合物划分（确保同一复合物的不同突变在同一折中）
    complex_ids = data['#Pdb_origin'].unique()
    print(f"唯一复合物数量: {len(complex_ids)}")
    
    # 初始化KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=config['random_seed'])
    
    # 存储结果
    fold_results = []
    
    print("\n开始五折交叉验证...")
    print(f"设备: {config['device']}")
    print(f"批大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"输出目录: {output_dir}")
    
    for fold, (train_complex_idx, val_complex_idx) in enumerate(kf.split(complex_ids)):
        train_complexes = complex_ids[train_complex_idx]
        val_complexes = complex_ids[val_complex_idx]
        
        # 划分数据
        train_data = data[data['#Pdb_origin'].isin(train_complexes)]
        val_data = data[data['#Pdb_origin'].isin(val_complexes)]
        
        print(f"\n折 {fold+1}:")
        print(f"  训练集: {len(train_data)} 突变, {len(train_complexes)} 复合物")
        print(f"  验证集: {len(val_data)} 突变, {len(val_complexes)} 复合物")
        
        # 训练当前折
        fold_result = train_fold(fold, train_data, val_data, config, output_dir)
        fold_results.append(fold_result)
        
        print(f"  结果: RMSE={fold_result['final_metrics']['rmse']:.4f}, "
              f"R²={fold_result['final_metrics']['r2']:.4f}")
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("五折交叉验证完成！")
    print(f"{'='*60}")
    
    # 计算平均指标
    rmse_scores = [r['final_metrics']['rmse'] for r in fold_results]
    r2_scores = [r['final_metrics']['r2'] for r in fold_results]
    pearson_scores = [r['final_metrics']['pearson_corr'] for r in fold_results]
    
    summary = {
        'fold_results': fold_results,
        'mean_rmse': np.mean(rmse_scores),
        'std_rmse': np.std(rmse_scores),
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_pearson_corr': np.mean(pearson_scores),
        'std_pearson_corr': np.std(pearson_scores),
        'config': config
    }
    
    # 打印总结
    print(f"\n整体性能:")
    print(f"  RMSE: {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}")
    print(f"  R²:   {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
    print(f"  Pearson: {summary['mean_pearson_corr']:.4f} ± {summary['std_pearson_corr']:.4f}")
    
    print(f"\n各折详细结果:")
    for i, result in enumerate(fold_results, 1):
        print(f"  折 {i}: RMSE={result['final_metrics']['rmse']:.4f}, "
              f"R²={result['final_metrics']['r2']:.4f}, "
              f"Pearson={result['final_metrics']['pearson_corr']:.4f}, "
              f"训练集={result['train_size']}, 验证集={result['val_size']}")
    
    # 保存总结
    summary_path = os.path.join(output_dir, 'cross_validation_summary.json')
    with open(summary_path, 'w') as f:
        # 转换numpy类型为Python原生类型
        summary_serializable = {}
        for key, value in summary.items():
            if key == 'fold_results':
                # 处理嵌套字典
                fold_results_serializable = []
                for fold in value:
                    fold_serializable = {}
                    for k, v in fold.items():
                        if isinstance(v, dict):
                            fold_serializable[k] = {
                                k2: float(v2) if isinstance(v2, (np.floating, float)) else v2
                                for k2, v2 in v.items()
                            }
                        else:
                            fold_serializable[k] = v
                    fold_results_serializable.append(fold_serializable)
                summary_serializable[key] = fold_results_serializable
            elif isinstance(value, (np.floating, float)):
                summary_serializable[key] = float(value)
            elif isinstance(value, np.integer):
                summary_serializable[key] = int(value)
            elif isinstance(value, np.ndarray):
                summary_serializable[key] = value.tolist()
            else:
                summary_serializable[key] = value
        
        json.dump(summary_serializable, f, indent=2)
    
    print(f"\n详细结果已保存到: {summary_path}")
    
    return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CHYModel五折交叉验证训练')
    parser.add_argument('--data_path', type=str, required=True,
                       help='SKEMPI数据文件路径')
    parser.add_argument('--pdb_base_path', type=str, required=True,
                       help='PDB文件基础路径')
    parser.add_argument('--output_dir', type=str, default='./chymodel_results',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='默认批大小')
    parser.add_argument('--train_batch_size', type=int, default=32,
                       help='训练批大小')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                       help='评估批大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--use_dummy_features', action='store_true',
                       help='使用虚拟特征（用于测试）')
    parser.add_argument('--no_geometric_features', action='store_true',
                       help='不使用几何特征')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'batch_size': args.batch_size,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'device': args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu',
        'pdb_base_path': args.pdb_base_path,
        'use_dummy_features': args.use_dummy_features,
        'use_geometric_features': not args.no_geometric_features
    }
    
    # 运行五折交叉验证
    results = five_fold_cross_validation(
        data_path=args.data_path,
        pdb_base_path=args.pdb_base_path,
        output_dir=args.output_dir,
        config=config
    )
    
    print(f"\n训练完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
