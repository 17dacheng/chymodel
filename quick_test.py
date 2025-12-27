#!/usr/bin/env python3
"""
快速测试真实数据
"""
import sys
import torch
from pathlib import Path

# 添加当前目录到路径
sys.path.append('/home/chengwang/code/chymodel')

def test_real_data():
    """测试真实数据"""
    print("测试真实数据处理...")
    
    try:
        from model_wo_esm_foldx import DDGModelTesterGeometric
        
        # 创建测试器
        tester = DDGModelTesterGeometric(
            pdb_base_path="/home/chengwang/data/SKEMPI/PDBs_fixed",
            cache_dir="./quick_test_cache"
        )
        
        # 测试CSV中的第一个样本
        csv_path = "/home/chengwang/code/chymodel/s1131.csv"
        import pandas as pd
        df = pd.read_csv(csv_path, sep='\t')
        
        # 获取第一个样本
        first_sample = df.iloc[0]
        pdb_id = first_sample['#Pdb_origin']
        mutation_str = first_sample['Mutation(s)_cleaned'].strip()
        
        # 解析突变
        chain = mutation_str[1]
        mutation = mutation_str[0] + mutation_str[2:]
        
        print(f"测试样本: PDB={pdb_id}, 链={chain}, 突变={mutation}")
        
        # 测试单个突变
        result = tester.test_single_mutation(pdb_id, chain, mutation)
        
        print(f"✓ 测试成功！")
        print(f"  - 预测ΔΔG: {result['predicted_ddg']:.3f} kcal/mol")
        print(f"  - 状态: {result['status']}")
        print(f"  - 时间: {result['total_time']:.3f}秒")
        
        return result['status'] == 'success'
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_data()
    sys.exit(0 if success else 1)