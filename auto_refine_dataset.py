import sys
from pathlib import Path
import argparse

# 执行指令
# uv run auto_refine_dataset.py -i "imudata/IDOL/idol_formatted" -o "imudata/IDOL/final_data"

# 引入路径
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
sys.path.append(str(project_root))

from StandardAdapter import StandardAdapter
from StandardEvaluator import StandardEvaluator

def main():
    parser = argparse.ArgumentParser(description="Refine and Calibrate Standard Datasets")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input folder (e.g., processed_data)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output folder (e.g., final_data)")
    args = parser.parse_args()
    
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    if not input_root.exists():
        print(f"Error: Input path {input_root} does not exist.")
        return

    # 递归查找包含 imu.csv 的文件夹
    # 假设结构是 input/seq_name/imu.csv
    seq_dirs = [p.parent for p in input_root.rglob("imu.csv")]
    
    print(f"Found {len(seq_dirs)} sequences.")
    
    for seq_dir in seq_dirs:
        try:
            # 保持相对目录结构
            rel_path = seq_dir.relative_to(input_root)
            target_dir = output_root / rel_path.parent
            
            # 1. 加载
            unit = StandardAdapter.load(seq_dir)
            
            # 2. 评估 & 校准 & 保存
            evaluator = StandardEvaluator(unit)
            evaluator.run_and_calibrate(output_root=target_dir)
            
        except Exception as e:
            print(f"❌ Failed to process {seq_dir.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()