#!/usr/bin/env python3
import torch
from model import DDGModelTester

def test_problematic_samples():
    """测试原始问题的样本"""
    print('=== 测试原始问题样本 ===')

    # 初始化测试器
    tester = DDGModelTester(
        pdb_base_path='/home/chengwang/data/SKEMPI/PDBs_fixed',
        cache_dir='./test_cache',
        use_geometric=True
    )

    # 测试问题样本
    problematic_samples = ['1FC2_C_Y10W', '1XD3_B_R72L', '1FC2_C_K31A']

    for sample in problematic_samples:
        print(f'\n测试样本: {sample}')
        try:
            parts = sample.split('_')
            pdb_id = parts[0]
            chain = parts[1]
            mutation = parts[2] if len(parts) > 2 else ''
            
            print(f'  PDB: {pdb_id}, Chain: {chain}, Mutation: {mutation}')
            
            # 提取几何特征
            wt_graph, mt_graph = tester.extract_geometric_features(pdb_id, chain, mutation)
            
            if wt_graph is not None and mt_graph is not None:
                print(f'  ✓ 特征提取成功')
                print(f'    WT图: {wt_graph.node_features.shape[0]}个原子')
                print(f'    MT图: {mt_graph.node_features.shape[0]}个原子')
                
                # 统计残基数量
                wt_residues = len(set(wt_graph.residue_indices))
                mt_residues = len(set(mt_graph.residue_indices))
                print(f'    WT残基数: {wt_residues}')
                print(f'    MT残基数: {mt_residues}')
                
                # 测试GNN处理
                try:
                    with torch.no_grad():
                        wt_geom = tester.model.geometric_gnn_wt(wt_graph)
                        mt_geom = tester.model.geometric_gnn_mt(mt_graph)
                        print(f'  ✓ GNN处理成功: WT={wt_geom.shape}, MT={mt_geom.shape}')
                        
                        # 显示显存使用情况
                        if torch.cuda.is_available():
                            print(f'    GPU显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
                            
                except Exception as e:
                    print(f'  ✗ GNN处理失败: {e}')
                    if "out of memory" in str(e).lower():
                        print(f'    显存不足！')
            else:
                print(f'  ✗ 特征提取失败')
                
        except Exception as e:
            print(f'  ✗ 测试失败: {e}')

    print('\n=== 测试完成 ===')

if __name__ == "__main__":
    test_problematic_samples()