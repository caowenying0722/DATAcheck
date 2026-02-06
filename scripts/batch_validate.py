#!/usr/bin/env python3
"""
批量验证所有模型和数据集

自动测试所有可用模型在所有数据集上的性能，生成完整的评估报告。

用法:
    python batch_validate.py [--parallel] [--workers N]

参数:
    --parallel: 启用并行执行模式（默认）
    --sequential: 使用顺序执行模式
    --workers N: 并行工作线程数（默认为CPU核心数）

输出:
    - results/batch_validation.json: 汇总所有测试结果
    - results/<model_name>_<device_name>/: 各个模型的具体结果
"""

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# 配置
RESULTS_PATH = Path("results")
DATASETS = [
    "~/Resources/VaildVersion2/ABR-AL60",
    "~/Resources/VaildVersion2/Redmi-K30-Pro",
]
MODELS_PATH = Path("models")

# 可用模型列表
AVAILABLE_MODELS = [
    "model_lstm_0105",
    "model_imot_0106",
    "model_imot_0111_64",
    "model_resnet_0111_96",
]

DATASETS_PATH = [Path(p).expanduser() for p in DATASETS]


def run_validation(
    model: str, dataset_path: Path, verbose: bool = True
) -> Dict[str, Any]:
    """运行单个模型在单个数据集上的验证"""
    result = {
        "model": model,
        "dataset": str(dataset_path),
        "success": False,
        "error": None,
        "output": None,
        "result_path": None,
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"测试模型: {model} | 数据集: {dataset_path.name}")
        print(f"{'=' * 50}")

    cmd = [
        sys.executable,
        "VaildModel.py",
        "-d",
        str(dataset_path),
        "-m",
        model,
        "--models_path",
        str(MODELS_PATH),
    ]

    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
        )

        result["output"] = process.stdout
        result["error"] = process.stderr
        result["success"] = process.returncode == 0

        if result["success"]:
            result["result_path"] = str(RESULTS_PATH / f"{model}_{dataset_path.name}")
            if verbose:
                print(f"✅ 成功: {model} on {dataset_path.name}")
        else:
            if verbose:
                print(f"❌ 失败: {model} on {dataset_path.name}")
                if process.stderr:
                    print(f"错误信息: {process.stderr[:200]}...")

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout after 300 seconds"
        if verbose:
            print(f"⏰ 超时: {model} on {dataset_path.name}")
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"💥 异常: {model} on {dataset_path.name} - {e}")

    return result


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成汇总报告"""
    summary = {
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r["success"]),
        "failed_tests": sum(1 for r in results if not r["success"]),
        "models": {},
        "datasets": {},
        "results": results,
    }

    # 按模型统计
    for model in AVAILABLE_MODELS:
        model_results = [r for r in results if r["model"] == model]
        summary["models"][model] = {
            "total": len(model_results),
            "successful": sum(1 for r in model_results if r["success"]),
            "failed": sum(1 for r in model_results if not r["success"]),
        }

    # 按数据集统计
    for dataset_path in DATASETS_PATH:
        dataset_results = [r for r in results if r["dataset"] == str(dataset_path)]
        summary["datasets"][dataset_path.name] = {
            "total": len(dataset_results),
            "successful": sum(1 for r in dataset_results if r["success"]),
            "failed": sum(1 for r in dataset_results if not r["success"]),
        }

    return summary


def run_validations_parallel(
    models: List[str], datasets: List[Path], max_workers: int | None = None
) -> List[Dict[str, Any]]:
    """并行运行所有验证任务"""
    results = []
    total_tests = len(models) * len(datasets)
    completed_tests = 0

    # 用于线程安全的打印
    print_lock = threading.Lock()

    def update_progress(result: Dict[str, Any]):
        nonlocal completed_tests
        with print_lock:
            completed_tests += 1
            status = "✅" if result["success"] else "❌"
            print(
                f"[{completed_tests}/{total_tests}] {status} {result['model']} on {Path(result['dataset']).name}"
            )

    # 创建所有任务
    tasks = [(model, dataset) for model in models for dataset in datasets]

    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(run_validation, model, dataset, verbose=False): (
                model,
                dataset,
            )
            for model, dataset in tasks
        }

        # 收集结果
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                update_progress(result)
            except Exception as e:
                model, dataset = future_to_task[future]
                error_result = {
                    "model": model,
                    "dataset": str(dataset),
                    "success": False,
                    "error": f"Execution error: {str(e)}",
                    "output": None,
                    "result_path": None,
                }
                results.append(error_result)
                update_progress(error_result)

    return results


def run_validations_sequential(
    models: List[str], datasets: List[Path]
) -> List[Dict[str, Any]]:
    """顺序运行所有验证任务"""
    results = []
    total_tests = len(models) * len(datasets)
    current_test = 0

    for model in models:
        for dataset_path in datasets:
            current_test += 1
            print(f"\n🔄 进度: {current_test}/{total_tests}")
            result = run_validation(model, dataset_path, verbose=True)
            results.append(result)

    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量验证所有模型和数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        default=True,
        help="启用并行执行模式（默认）",
    )
    parser.add_argument(
        "--sequential", "-s", action="store_true", help="使用顺序执行模式"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="并行工作线程数（默认为CPU核心数）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 如果明确指定了 sequential，则禁用并行
    use_parallel = args.parallel and not args.sequential

    print("🚀 开始批量验证所有模型和数据集")
    print(f"模型路径: {MODELS_PATH}")
    print(f"数据集路径: {DATASETS_PATH}")

    # 检查环境
    if not MODELS_PATH.exists():
        print(f"❌ 模型路径不存在: {MODELS_PATH}")
        return

    # 检查数据集
    valid_datasets = []
    for dataset_path in DATASETS_PATH:
        if dataset_path.exists():
            valid_datasets.append(dataset_path)
        else:
            print(f"⚠️  数据集不存在: {dataset_path}")

    if not valid_datasets:
        print("❌ 未找到可用数据集")
        return

    print(f"📋 可用模型: {len(AVAILABLE_MODELS)} 个")
    for model in AVAILABLE_MODELS:
        model_file = MODELS_PATH / f"{model}.pt"
        status = "✅" if model_file.exists() else "❌"
        print(f"  {status} {model}")

    print(f"📊 可用数据集: {len(valid_datasets)} 个")
    for dataset_path in valid_datasets:
        print(f"  📁 {dataset_path.name}")

    # 自动继续执行
    print("\n🔄 开始执行所有测试...")

    if use_parallel:
        print(
            f"⚡️ 使用并行模式执行测试（最多 {args.workers or 'CPU核心数'} 个并行任务）\n"
        )
        all_results = run_validations_parallel(
            models=AVAILABLE_MODELS, datasets=valid_datasets, max_workers=args.workers
        )
    else:
        print("📝 使用顺序模式执行测试\n")
        all_results = run_validations_sequential(
            models=AVAILABLE_MODELS, datasets=valid_datasets
        )

    # 生成汇总报告
    print("\n📈 生成汇总报告...")
    summary = generate_summary_report(all_results)

    # 保存结果
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_PATH / "batch_validation.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f"\n{'=' * 60}")
    print("🎉 批量验证完成!")
    print(f"{'=' * 60}")
    print(f"总测试数: {summary['total_tests']}")
    print(f"成功: {summary['successful_tests']} ✅")
    print(f"失败: {summary['failed_tests']} ❌")
    print(f"成功率: {summary['successful_tests'] / summary['total_tests'] * 100:.1f}%")
    print(f"详细报告: {report_path}")

    # 按模型显示结果
    print("\n📊 按模型统计:")
    for model, stats in summary["models"].items():
        success_rate = (
            stats["successful"] / stats["total"] * 100 if stats["total"] > 0 else 0
        )
        print(
            f"  {model}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)"
        )

    # 按数据集显示结果
    print("\n📊 按数据集统计:")
    for dataset, stats in summary["datasets"].items():
        success_rate = (
            stats["successful"] / stats["total"] * 100 if stats["total"] > 0 else 0
        )
        print(
            f"  {dataset}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)"
        )


if __name__ == "__main__":
    main()
