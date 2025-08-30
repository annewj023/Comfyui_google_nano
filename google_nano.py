import os
import io
import base64
import string
import traceback
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# 可选：批量模式支持 XLSX 需要 pandas 和 openpyxl
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # 延迟报错，在调用时提示安装依赖


def _pil_to_base64_data_url(img: Image.Image, format: str = "jpeg") -> str:
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=format)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format};base64,{img_str}"


def _decode_image_from_openrouter_response(completion) -> Tuple[List[Image.Image], str]:
    """
    解析 OpenRouter chat.completions 响应中的 base64 图片，返回 PIL 列表或错误信息。
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


def _tensor_to_pils(image) -> List[Image.Image]:
    """
    将 ComfyUI 的 IMAGE(tensor[B,H,W,3], 浮点0-1) 转成 PIL 列表
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


def _pils_to_tensor(pils: List[Image.Image]) -> torch.Tensor:
    """
    将 PIL 列表转回 ComfyUI 的 IMAGE tensor[B,H,W,3], float32 0-1
    如果图片尺寸不一致，则分别处理每张图片，不强制统一尺寸
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


# 1. 修改类名，确保它在Python中是唯一的
class GoogleNanoNode:
    """
    使用 OpenRouter Chat Completions，通过单条 prompt 或 CSV/Excel 批量，根据输入参考图生成新图。
    - 单图：提供 prompt
    - 批量：提供 file_path（含 'prompt' 列）
    输出：
      IMAGE: 生成的图像（单张或批量拼成 batch）
      STRING: 状态/日志
    """

    CATEGORY = "OpenRouter"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "file_path": ("STRING", {"multiline": False, "default": ""}),
                "site_url": ("STRING", {"multiline": False, "default": ""}),
                "site_name": ("STRING", {"multiline": False, "default": ""}),
                "model": ("STRING", {"multiline": False, "default": "google/gemini-2.5-flash-image-preview:free"}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }

    def _call_openrouter(
        self,
        api_key: str,
        pil_refs: List[Image.Image],
        prompt_text: str,
        site_url: str,
        site_name: str,
        model: str,
    ) -> Tuple[List[Image.Image], str]:
        if OpenAI is None:
            return [], "未安装 openai 库，请先安装：pip install openai"
        if not api_key:
            return [], "错误：请输入 OpenRouter API Key。"

        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            headers = {}
            if site_url:
                headers["HTTP-Referer"] = site_url
            if site_name:
                headers["X-Title"] = site_name

            if len(pil_refs) > 1:
                full_prompt = f"请严格根据这些图片，并结合以下提示词，生成一张新的图片。不要描述图片。提示词：'{prompt_text}'"
            else:
                full_prompt = f"请严格根据这张图片，并结合以下提示词，生成一张新的图片。不要描述图片。提示词：'{prompt_text}'"

            content = [{"type": "text", "text": full_prompt}]
            for pil_ref in pil_refs:
                data_url = _pil_to_base64_data_url(pil_ref, format="jpeg")
                content.append({"type": "image_url", "image_url": {"url": data_url}})

            completion = client.chat.completions.create(
                extra_headers=headers,
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            )
            pils, err = _decode_image_from_openrouter_response(completion)
            if err:
                return [], err
            if not pils:
                return [], "未从模型收到图片数据。"
            return pils, ""
        except Exception as e:
            return [], f"生成图片时出错: {traceback.format_exc()}"

    def generate(
        self,
        api_key: str,
        prompt: str = "",
        file_path: str = "",
        site_url: str = "",
        site_name: str = "",
        model: str = "google/gemini-2.5-flash-image-preview:free",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
    ):
        all_input_pils: List[Image.Image] = []
        try:
            for img_tensor in [image1, image2, image3, image4]:
                if img_tensor is not None:
                    all_input_pils.extend(_tensor_to_pils(img_tensor))
        except Exception as e:
            return (_pils_to_tensor([]), f"输入图像解析失败：{e}")

        if not all_input_pils:
            return (_pils_to_tensor([]), "错误：请输入至少一张参考图像。")
        
        # 不需要提前转换成tensor，直接使用PIL图片调用API

        # 判定模式
        if not prompt and not file_path:
            # 如果没有操作，返回原始输入图片的tensor
            return (_pils_to_tensor(all_input_pils), "错误：请输入提示词或提供 CSV/Excel 文件路径。")

        all_out_pils: List[Image.Image] = []
        status_msgs: List[str] = []

        # 单条 prompt
        if prompt:
            out_pils, err = self._call_openrouter(api_key, all_input_pils, prompt, site_url, site_name, model)
            if err:
                # 出错时返回原始输入图片
                return (_pils_to_tensor(all_input_pils), err)
            all_out_pils.extend(out_pils)
            status_msgs.append(f"已生成 {len(out_pils)} 张图片。")

        # 批量文件
        elif file_path:
            # 改进的路径处理，支持中文路径和带引号的路径
            clean_path = file_path.strip()
            
            # 移除路径两端的引号（支持单引号和双引号）
            if (clean_path.startswith('"') and clean_path.endswith('"')) or \
               (clean_path.startswith("'") and clean_path.endswith("'")):
                clean_path = clean_path[1:-1]
            
            # 处理路径中的特殊字符，但保留中文字符
            # 只移除不可打印的控制字符，保留中文等Unicode字符
            import re
            # 移除控制字符但保留正常的Unicode字符（包括中文）
            clean_path = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_path)
            
            # 标准化路径分隔符（Windows兼容性）
            clean_path = os.path.normpath(clean_path)
            
            if not os.path.exists(clean_path):
                return (_pils_to_tensor(all_input_pils), f"错误：文件路径不存在: {clean_path}\n请检查路径是否正确，支持中文路径和带空格的路径。")

            if not HAS_PANDAS:
                return (_pils_to_tensor(all_input_pils), "错误：批量模式需要 pandas，请先安装：pip install pandas openpyxl")

            try:
                if clean_path.lower().endswith(".csv"):
                    # 使用UTF-8编码读取CSV文件，支持中文
                    try:
                        df = pd.read_csv(clean_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        # 如果UTF-8失败，尝试GBK编码（中文Windows系统常用）
                        try:
                            df = pd.read_csv(clean_path, encoding='gbk')
                        except UnicodeDecodeError:
                            # 最后尝试自动检测编码
                            df = pd.read_csv(clean_path, encoding='latin1')
                else:
                    # 读取Excel文件
                    df = pd.read_excel(clean_path, sheet_name="Sheet1")
            except Exception as e:
                return (_pils_to_tensor(all_input_pils), f"读取文件失败：{e}\n请确认：\n1. 文件格式是否正确（CSV或Excel）\n2. 文件是否被其他程序占用\n3. 文件编码是否正确（建议UTF-8）")

            if "prompt" not in df.columns:
                return (_pils_to_tensor(all_input_pils), "错误：文件中未找到 'prompt' 列。")

            for idx, row in df.iterrows():
                csv_prompt = row.get("prompt")
                if not isinstance(csv_prompt, str) or not csv_prompt.strip():
                    status_msgs.append(f"第 {idx + 1} 行跳过：空提示词")
                    continue
                out_pils, err = self._call_openrouter(api_key, all_input_pils, csv_prompt, site_url, site_name, model)
                if err:
                    status_msgs.append(f"图片 {idx + 1} 生成失败：{err}")
                else:
                    all_out_pils.extend(out_pils)
                    status_msgs.append(f"图片 {idx + 1} 生成成功（{len(out_pils)} 张）。")

            if not all_out_pils:
                return (_pils_to_tensor(all_input_pils), "未从文件中生成任何图片。\n" + "\n".join(status_msgs))

        # 只在最后处理输出结果时才需要转换成tensor
        out_tensor = _pils_to_tensor(all_out_pils)
        
        # 检查是否有多张不同尺寸的图片，如果有，添加说明
        if len(all_out_pils) > 1:
            sizes = [(pil.width, pil.height) for pil in all_out_pils]
            unique_sizes = list(set(sizes))
            if len(unique_sizes) > 1:
                size_info = f"\n注意：生成了 {len(all_out_pils)} 张不同尺寸的图片 {unique_sizes}，ComfyUI只显示第一张。"
                status = ("\n".join(status_msgs) + size_info) if status_msgs else ("完成" + size_info)
            else:
                status = "\n".join(status_msgs) if status_msgs else "完成"
        else:
            status = "\n".join(status_msgs) if status_msgs else "完成"
            
        return (out_tensor, status)


# 注册到 ComfyUI
# 2. 修改节点类映射，使用新的类名作为键和值
NODE_CLASS_MAPPINGS = {
    "GoogleNanoNode": GoogleNanoNode,
}
# 3. 修改节点显示名称映射，使用新的类名作为键
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleNanoNode": "google nano",
}
