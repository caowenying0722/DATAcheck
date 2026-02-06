"""
作者: qi-xmu
功能: TorchScript模型转Android兼容格式工具
描述: 该脚本用于将PyTorch的TorchScript模型转换为Android平台可使用的格式，
      支持模型优化和量化功能以提高在移动设备上的推理性能。
用法:
    python TorchScript2Android.py -i input_model.pt -o output_model.pt
"""

import torch
import torch.utils.mobile_optimizer as mobile_optimizer


def convert_torchscript_to_android(input_model_path, output_model_path, quantize=False):
    """
    将TorchScript模型转换为Android兼容格式

    参数:
        input_model_path: 输入的TorchScript模型路径(.pt)
        output_model_path: 输出的Android模型路径(.pt)
        quantize: 是否进行量化(默认False)
    """
    # 1. 加载TorchScript模型(强制使用CPU)
    device = torch.device("cpu")
    model = torch.jit.load(input_model_path, map_location=device)
    model.eval()

    # 2. 模型优化
    optimized_model = mobile_optimizer.optimize_for_mobile(model)

    # 3. 可选量化
    if quantize:
        # 使用动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            optimized_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        optimized_model = quantized_model

    # 4. 保存优化后的模型
    optimized_model._save_for_lite_interpreter(output_model_path)
    print(f"模型已成功转换并保存到 {output_model_path}")


if __name__ == "__main__":
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="将模型转换为Android兼容格式")
    parser.add_argument("-i", "--input", required=True, help="输入TorchScript模型(.pt)")
    parser.add_argument("-o", "--output", required=True, help="输出Android模型(.pt)")
    parser.add_argument("-q", "--quantize", action="store_true", help="是否进行量化")

    # 解析参数
    args = parser.parse_args()
    # 调用转换函数
    convert_torchscript_to_android(args.input, args.output, quantize=args.quantize)
