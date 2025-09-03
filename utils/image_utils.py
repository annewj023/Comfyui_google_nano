"""
图像处理工具函数 - 从原google_nano.py迁移的图像处理功能
"""

import io
import os
import base64
import traceback
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


def pil_to_base64_data_url(img: Image.Image, format: str = "jpeg") -> str:
    """
    将PIL图像转换为base64数据URL
    
    Args:
        img: PIL图像对象
        format: 图像格式
        
    Returns:
        base64数据URL字符串
    """
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=format)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format};base64,{img_str}"


def decode_image_from_openrouter_response(completion) -> Tuple[List[Image.Image], str]:
    """
    解析 OpenRouter chat.completions 响应中的 base64 图片，返回 PIL 列表或错误信息。
    
    Args:
        completion: OpenRouter API响应对象
        
    Returns:
        Tuple[List[Image.Image], str]: (图像列表, 错误信息)
    """
    try:
        response_dict = completion.model_dump()
        images_list = response_dict.get("choices", [{}])[0].get("message", {}).get("images")
        if images_list and isinstance(images_list, list) and len(images_list) > 0:
            out_pils = []
            for image_info in images_list:
                base64_url = image_info.get("image_url", {}).get("url")
                if not base64_url:
                    continue
                # 支持 data URL 或纯 base64
                if "base64," in base64_url:
                    base64_data = base64_url.split("base64,")[1]
                else:
                    base64_data = base64_url
                img_bytes = base64.b64decode(base64_data)
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                out_pils.append(pil)
            if out_pils:
                return out_pils, ""
        # 未取到图片，回显原始 JSON
        return [], f"模型回复中未直接包含图片数据。\n\n--- 完整的API回复 ---\n{completion.model_dump_json(indent=2)}"
    except Exception as e:
        try:
            raw = completion.model_dump_json(indent=2)
        except Exception:
            raw = "<failed to dump json>"
        return [], f"解析API响应时出错: {e}\n\n--- 完整的API回复 ---\n{raw}"


def tensor_to_pils(image) -> List[Image.Image]:
    """
    将 ComfyUI 的 IMAGE(tensor[B,H,W,3], 浮点0-1) 转成 PIL 列表
    
    Args:
        image: ComfyUI IMAGE张量或包含images键的字典
        
    Returns:
        PIL图像列表
    """
    if isinstance(image, dict) and "images" in image:
        image = image["images"]
    if not isinstance(image, torch.Tensor):
        raise TypeError("IMAGE 输入应为 torch.Tensor 或包含 'images' 键的 dict")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    imgs = []
    arr = (image.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)  # [B,H,W,3]
    for i in range(arr.shape[0]):
        pil = Image.fromarray(arr[i], mode="RGB")
        imgs.append(pil)
    return imgs


def pils_to_tensor(pils: List[Image.Image]) -> torch.Tensor:
    """
    将 PIL 列表转回 ComfyUI 的 IMAGE tensor[B,H,W,3], float32 0-1
    如果图片尺寸不一致，则分别处理每张图片，不强制统一尺寸
    
    Args:
        pils: PIL图像列表
        
    Returns:
        ComfyUI IMAGE张量
    """
    if not pils:
        # 返回一个空的占位张量，避免下游崩溃（B=0）
        return torch.zeros((0, 64, 64, 3), dtype=torch.float32)
    
    # 如果只有一张图片，直接处理
    if len(pils) == 1:
        pil = pils[0]
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)  # [H,W,3]
        return tensor.unsqueeze(0)  # [1,H,W,3]
    
    # 检查所有图片是否具有相同尺寸
    first_size = (pils[0].width, pils[0].height)
    all_same_size = all((pil.width, pil.height) == first_size for pil in pils)
    
    if all_same_size:
        # 所有图片尺寸相同，可以直接堆叠
        np_imgs = []
        for pil in pils:
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
            np_imgs.append(arr)
        batch = np.stack(np_imgs, axis=0).astype(np.float32) / 255.0  # [B,H,W,3]
        return torch.from_numpy(batch)
    else:
        # 图片尺寸不同，只返回第一张图片，并在状态中说明
        # 这是ComfyUI的限制：IMAGE类型要求batch中所有图片尺寸相同
        pil = pils[0]
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)  # [H,W,3]
        return tensor.unsqueeze(0)  # [1,H,W,3]


def validate_and_convert_images(image_tensors: List) -> List[Image.Image]:
    """
    验证并转换输入的图像张量为PIL图像列表
    
    Args:
        image_tensors: 图像张量列表
        
    Returns:
        PIL图像列表
        
    Raises:
        ValueError: 图像转换失败时抛出
    """
    all_pils = []
    try:
        for img_tensor in image_tensors:
            if img_tensor is not None:
                pils = tensor_to_pils(img_tensor)
                all_pils.extend(pils)
    except Exception as e:
        raise ValueError(f"输入图像解析失败：{e}")
    
    if not all_pils:
        raise ValueError("错误：请输入至少一张参考图像。")
    
    return all_pils


def create_size_mismatch_message(pils: List[Image.Image]) -> str:
    """
    创建图片尺寸不匹配的提示信息
    
    Args:
        pils: PIL图像列表
        
    Returns:
        尺寸信息字符串
    """
    if len(pils) > 1:
        sizes = [(pil.width, pil.height) for pil in pils]
        unique_sizes = list(set(sizes))
        if len(unique_sizes) > 1:
            return (f"\n⚠️ 尺寸不匹配警告：生成了 {len(pils)} 张不同尺寸的图片 {unique_sizes}\n"
                   f"由于ComfyUI限制，只能显示第一张图片。\n"
                   f"💾 所有图片已自动保存到ComfyUI输出目录中的google_nano文件夹。\n"
                   f"💡 建议：在提示词中明确指定尺寸要求以获得相同尺寸的图片。")
    return ""


def get_actual_display_count(pils: List[Image.Image]) -> int:
    """
    获取实际显示的图片数量（考虑ComfyUI尺寸限制）
    
    Args:
        pils: PIL图像列表
        
    Returns:
        实际显示的图片数量
    """
    if not pils:
        return 0
    
    if len(pils) == 1:
        return 1
    
    # 检查所有图片是否具有相同尺寸
    first_size = (pils[0].width, pils[0].height)
    all_same_size = all((pil.width, pil.height) == first_size for pil in pils)
    
    if all_same_size:
        return len(pils)  # 所有图片都可以显示
    else:
        return 1  # 只能显示第一张图片
def save_images_to_output(pils: List[Image.Image], task_id: str = None, prompt: str = "") -> List[str]:
    """
    将生成的图片保存到ComfyUI输出目录
    
    Args:
        pils: PIL图像列表
        task_id: 任务ID（可选）
        prompt: 提示词（可选）
        
    Returns:
        保存的文件路径列表
    """
    if not pils:
        return []
    
    saved_paths = []
    
    try:
        # 获取ComfyUI输出目录
        output_dir = get_comfyui_output_dir()
        
        # 创建子目录（按日期组织）
        today = datetime.now().strftime("%Y%m%d")
        sub_dir = os.path.join(output_dir, "google_nano", today)
        os.makedirs(sub_dir, exist_ok=True)
        
        # 生成文件名前缀
        timestamp = datetime.now().strftime("%H%M%S")
        if task_id:
            prefix = f"task_{task_id}_{timestamp}"
        else:
            prefix = f"google_nano_{timestamp}"
        
        # 保存每张图片
        for i, pil in enumerate(pils):
            # 生成文件名
            if len(pils) == 1:
                filename = f"{prefix}.png"
            else:
                filename = f"{prefix}_{i+1:02d}.png"
            
            file_path = os.path.join(sub_dir, filename)
            
            # 保存图片
            pil.save(file_path, "PNG", optimize=True)
            saved_paths.append(file_path)
            
        print(f"[INFO] 已保存 {len(saved_paths)} 张图片到: {sub_dir}")
        
    except Exception as e:
        print(f"[ERROR] 保存图片失败: {e}")
        import traceback
        traceback.print_exc()
    
    return saved_paths


def get_comfyui_output_dir() -> str:
    """
    获取ComfyUI输出目录
    
    Returns:
        ComfyUI输出目录路径
    """
    try:
        # 尝试使用ComfyUI的folder_paths模块
        try:
            import folder_paths
            return folder_paths.get_output_directory()
        except ImportError:
            pass
        
        # 如果无法导入folder_paths，尝试查找常见的ComfyUI路径
        possible_paths = [
            "output",  # 相对路径
            "../../../output",  # 从 custom_nodes 向上查找
            "../../../../output",  # 更深层级
            os.path.expanduser("~/ComfyUI/output"),  # 用户目录
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) or path == "output":
                # 确保目录存在
                os.makedirs(abs_path, exist_ok=True)
                return abs_path
        
        # 如果都找不到，创建默认输出目录
        default_output = os.path.abspath("output")
        os.makedirs(default_output, exist_ok=True)
        return default_output
        
    except Exception as e:
        print(f"[WARNING] 获取输出目录失败: {e}，使用当前目录")
        return os.getcwd()