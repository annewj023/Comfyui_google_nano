import os
import io
import base64
import string
import traceback
import uuid
import time
import threading
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

# 导入新的管理器和工具类
from .managers import ConfigManager, ApiKeyManager, TaskLogger
from .managers.task_logger import TaskStatus
from .utils import (
    pil_to_base64_data_url, decode_image_from_openrouter_response,
    tensor_to_pils, pils_to_tensor, validate_and_convert_images,
    create_size_mismatch_message, get_actual_display_count, save_images_to_output,
    retry_with_backoff, ApiCallError
)

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


# 移除旧的工具函数，它们已经迁移到utils模块


# 1. 修改类名，确保它在Python中是唯一的
class GoogleNanoNode:
    """
    使用 OpenRouter Chat Completions，通过单条 prompt 或 CSV/Excel 批量，根据输入参考图生成新图。
    
    新功能：
    - 并发图片生成数量控制（1-10范围）
    - 多API Key管理与调度（轮换模式和并行模式）
    - API Key状态监控与自动排除
    - 模型选择功能
    - 配置文件架构调整（config.json）
    - 失败重试机制和详细日志记录
    
    输出：
      IMAGE: 生成的图像（单张或批量拼成 batch）
      STRING: 状态/日志
    """

    CATEGORY = "OpenRouter"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")  # 添加第三个返回值用于Key状态显示
    RETURN_NAMES = ("image", "status", "key_status")  # 为输出命名
    OUTPUT_NODE = False
    
    # 类级别的管理器实例（单例模式）
    _config_manager = None
    _api_key_manager = None
    _task_logger = None
    _manager_lock = threading.Lock()

    @classmethod
    def _get_managers(cls):
        """获取管理器实例（单例模式）"""
        if cls._config_manager is None:
            with cls._manager_lock:
                if cls._config_manager is None:
                    try:
                        print("[DEBUG] 初始化ConfigManager...")
                        cls._config_manager = ConfigManager()
                        print("[DEBUG] ConfigManager初始化成功")
                        
                        print("[DEBUG] 初始化ApiKeyManager...")
                        cls._api_key_manager = ApiKeyManager(cls._config_manager)
                        print("[DEBUG] ApiKeyManager初始化成功")
                        
                        print("[DEBUG] 初始化TaskLogger...")
                        log_level = cls._config_manager.get_setting("log_level", "INFO")
                        cls._task_logger = TaskLogger(log_level=log_level)
                        print("[DEBUG] TaskLogger初始化成功")
                        
                    except Exception as e:
                        print(f"[ERROR] 管理器初始化失败: {e}")
                        import traceback
                        traceback.print_exc()
                        # 返回None让调用者处理
                        return None, None, None
                        
        # 验证所有管理器都正确初始化
        if not all([cls._config_manager, cls._api_key_manager, cls._task_logger]):
            print(f"[ERROR] 管理器初始化不完整: config={cls._config_manager}, api={cls._api_key_manager}, logger={cls._task_logger}")
            return None, None, None
            
        return cls._config_manager, cls._api_key_manager, cls._task_logger

    @classmethod
    def INPUT_TYPES(cls):
        # 获取配置管理器以读取模型列表和API Key状态
        try:
            config_manager, api_key_manager, _ = cls._get_managers()
            available_models = config_manager.get_models()
            api_keys_status = api_key_manager.get_key_statistics()

            # 调度模式选项
            scheduling_modes = ["round_robin", "random", "weighted"]

            # 获取已配置的API Keys信息用于显示
            configured_keys = config_manager.get_api_keys()
            key_count = len(configured_keys)

            # 获取配置中的key_management_mode默认值
            default_key_management_mode = config_manager.get_setting("key_management_mode", "同时使用两者")

        except Exception as e:
            # 如果配置加载失败，使用默认值
            available_models = [
                "google/gemini-2.5-flash-image-preview:free",
                "google/gemini-2.5-flash-image-preview"
            ]
            scheduling_modes = ["round_robin"]
            key_count = 0
            api_keys_status = {"total_keys": 0, "available_keys": 0, "key_details": []}
            default_key_management_mode = "同时使用两者"
        
        # 创建动态的API Key输入字段
        api_key_inputs = {}
        
        # 主API Key（保持向后兼容）
        api_key_inputs["api_key_main"] = ("STRING", {
            "multiline": False, 
            "default": "",
            "tooltip": "主API Key（必填）"
        })
        
        # 动态添加额外的API Key字段（最多支持10个）
        max_keys = min(10, max(1, key_count + 2))  # 至少1个，最多10个，当前数量+2个备用
        
        for i in range(2, max_keys + 1):
            api_key_inputs[f"api_key_{i}"] = ("STRING", {
                "multiline": False, 
                "default": "",
                "tooltip": f"API Key {i}（可选）"
            })
        
        # API Key状态显示字段（只读信息）
        status_info = []
        if api_keys_status["key_details"]:
            for i, key_detail in enumerate(api_keys_status["key_details"][:5]):  # 最多显示5个Key状态
                try:
                    # 安全获取字段值，避免KeyError
                    status = key_detail.get('status', '未知')
                    remaining = key_detail.get('remaining', '未知')
                    success_rate = key_detail.get('success_rate', 0)
                    status_text = f"Key{i+1}: {status} - 剩余:{remaining} - 成功率:{success_rate}%"
                    status_info.append(status_text)
                except Exception as e:
                    status_info.append(f"Key{i+1}: 状态解析失败 - {e}")
        
        if not status_info:
            status_info = ["暂无API Key状态信息"]
        
        input_types = {
            "required": {
                # 主 API Key（必填）
                "api_key_main": api_key_inputs["api_key_main"],
                # 图像输入（至少需要一张）
                "image1": ("IMAGE",),
            },
            "optional": {
                # 额外的API Key输入
                **{k: v for k, v in api_key_inputs.items() if k != "api_key_main"},
                
                # 基础参数
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "tooltip": "单图生成的提示词"
                }),
                "file_path": ("STRING", {
                    "multiline": False, 
                    "default": "",
                    "tooltip": "批量处理的CSV/Excel文件路径"
                }),
                "site_url": ("STRING", {
                    "multiline": False, 
                    "default": "",
                    "tooltip": "网站URL（可选）"
                }),
                "site_name": ("STRING", {
                    "multiline": False, 
                    "default": "",
                    "tooltip": "网站名称（可选）"
                }),
                
                # 模型选择
                "model": (available_models, {
                    "default": available_models[0] if available_models else "google/gemini-2.5-flash-image-preview:free",
                    "tooltip": "选择要使用的AI模型"
                }),
                
                # 并发控制
                "max_concurrent": ("INT", {
                    "default": 3, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1,
                    "tooltip": "最大并发任务数量（1-10）"
                }),
                
                # 调度策略
                "scheduling_mode": (scheduling_modes, {
                    "default": "round_robin",
                    "tooltip": "API Key调度模式：轮换/随机/加权"
                }),
                
                # 并行模式
                "enable_parallel": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用并行模式（同时使用多个API Key）"
                }),
                
                # 重试设置
                "max_retries": ("INT", {
                    "default": 3, 
                    "min": 0, 
                    "max": 10, 
                    "step": 1,
                    "tooltip": "API调用失败时的最大重试次数"
                }),
                
                # 详细日志
                "enable_detailed_logs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用详细的任务执行日志"
                }),
                
                # Key管理模式
                "key_management_mode": (["\u4f7f\u7528\u8f93\u5165\u7684Key", "\u4f7f\u7528\u914d\u7f6e\u6587\u4ef6Key", "\u540c\u65f6\u4f7f\u7528\u4e24\u8005"], {
                    "default": default_key_management_mode,
                    "tooltip": "选择API Key使用模式"
                }),
                
                # 实时状态更新开关
                "auto_refresh_status": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自动刷新API Key状态信息"
                }),
                
                # 额外的图像输入
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }
        
        return input_types
    
    @classmethod
    def get_key_status_info(cls, key_management_mode: str = None) -> str:
        """获取当前API Key状态信息的格式化字符串"""
        try:
            managers = cls._get_managers()
            if managers is None or None in managers:
                return "管理器初始化失败"
                
            config_manager, api_key_manager, _ = managers
            if not config_manager or not api_key_manager:
                return "管理器不可用"
                
            stats = api_key_manager.get_key_statistics()
            
            if not stats["key_details"]:
                return "暂无API Key状态信息"
            
            status_lines = []

            # 根据管理模式添加说明
            if key_management_mode:
                status_lines.append(f"当前模式: {key_management_mode}")

            status_lines.append(f"总计: {stats['total_keys']}个 Key, 可用: {stats['available_keys']}个")
            
            for i, key_detail in enumerate(stats["key_details"][:5]):  # 最多显示5个
                status_icon = "✅" if key_detail['status'] == 'available' else "❌"
                
                # 构建状态文本
                status_text = f"{status_icon} Key{i+1}({key_detail['name']}): {key_detail['status']}"
                
                # 添加余额信息
                remaining = key_detail.get('remaining', '未知')
                if remaining != '未知':
                    if remaining == "unlimited":
                        status_text += f" | 余额:无限制"
                    else:
                        status_text += f" | 余额:{remaining}"
                
                # 添加使用情况
                usage = key_detail.get('usage', 0)
                if usage > 0:
                    status_text += f" | 已用:{usage}"
                
                # 添加成功率
                if key_detail['success_rate'] > 0:
                    status_text += f" | 成功率:{key_detail['success_rate']}%"
                
                # 添加免费层标识
                if key_detail.get('is_free_tier', False):
                    status_text += f" | 🆓免费层"
                
                status_lines.append(status_text)
            
            return "\n".join(status_lines)
            
        except Exception as e:
            return f"获取状态信息失败: {e}"
    
    @classmethod
    def debug_managers(cls):
        """调试管理器状态"""
        print("[DEBUG] 检查管理器状态...")
        print(f"[DEBUG] _config_manager: {cls._config_manager}")
        print(f"[DEBUG] _api_key_manager: {cls._api_key_manager}")
        print(f"[DEBUG] _task_logger: {cls._task_logger}")
        
        try:
            managers = cls._get_managers()
            print(f"[DEBUG] _get_managers() 返回: {managers}")
            if managers and len(managers) == 3:
                config_manager, api_key_manager, task_logger = managers
                print(f"[DEBUG] 解包后: config={config_manager}, api={api_key_manager}, logger={task_logger}")
        except Exception as e:
            print(f"[DEBUG] _get_managers() 异常: {e}")
            import traceback
            traceback.print_exc()
    
    @classmethod
    def refresh_key_status(cls):
        """刷新API Key状态信息"""
        try:
            config_manager, api_key_manager, _ = cls._get_managers()
            # 先清理过期冷却
            api_key_manager.cleanup_expired_cooldowns()
            # 然后刷新所有Key状态
            results = api_key_manager.refresh_all_keys_status()
            return results
        except Exception as e:
            print(f"[ERROR] 刷新Key状态失败: {e}")
            return {}

    def _call_openrouter(self, task_logger, log_id: str, api_key_config: Dict[str, Any], 
                        pil_refs: List[Image.Image], prompt_text: str, 
                        site_url: str, site_name: str, model: str) -> Tuple[List[Image.Image], str]:
        """
        调用OpenRouter API生成图片
        
        Args:
            task_logger: 日志记录器
            log_id: 任务日志ID
            api_key_config: API Key配置
            pil_refs: 参考图片列表
            prompt_text: 提示词
            site_url: 网站URL
            site_name: 网站名称
            model: 模型名称
            
        Returns:
            Tuple[List[Image.Image], str]: (生成的图片列表, 错误信息)
        """
        if OpenAI is None:
            error_msg = "未安装 openai 库，请先安装：pip install openai"
            task_logger.log_error(log_id, error_msg, "dependency")
            return [], error_msg
        
        api_key_value = api_key_config.get("value", "")
        api_key_id = api_key_config.get("id", "unknown")
        
        if not api_key_value:
            error_msg = "错误：API Key为空。"
            task_logger.log_error(log_id, error_msg, "configuration")
            return [], error_msg

        start_time = time.time()
        
        try:
            # 创建OpenAI客户端，设置超时以防止无限等待
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1", 
                api_key=api_key_value,
                timeout=30.0  # 30秒超时，防止网络请求阻塞UI
            )
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
                data_url = pil_to_base64_data_url(pil_ref, format="jpeg")
                content.append({"type": "image_url", "image_url": {"url": data_url}})

            # 记录请求数据
            request_data = {
                "prompt": prompt_text,
                "model": model,
                "images": pil_refs,
                "image_count": len(pil_refs)
            }

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
            
            duration = time.time() - start_time
            
            pils, err = decode_image_from_openrouter_response(completion)
            if err:
                task_logger.log_api_call(log_id, api_key_id, model, request_data, None, err, duration)
                return [], err
            if not pils:
                error_msg = "未从模型收到图片数据。"
                task_logger.log_api_call(log_id, api_key_id, model, request_data, None, error_msg, duration)
                return [], error_msg
            
            # 记录成功的API调用，包含实际显示数量
            actual_display_count = get_actual_display_count(pils)
            response_data = {
                "images_generated": len(pils),
                "images_displayed": actual_display_count,
                "images_info": [f"image_{i}" for i in range(len(pils))]
            }
            task_logger.log_api_call(log_id, api_key_id, model, request_data, response_data, None, duration)
            
            return pils, ""
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"生成图片时出错: {str(e)}"
            
            # 记录失败的API调用
            request_data = {
                "prompt": prompt_text,
                "model": model,
                "images": pil_refs,
                "image_count": len(pil_refs)
            }
            task_logger.log_api_call(log_id, api_key_id, model, request_data, None, error_msg, duration)
            
            return [], error_msg
    
    def _process_single_prompt(self, managers: Tuple, log_id: str, 
                              pil_refs: List[Image.Image], prompt_text: str,
                              site_url: str, site_name: str, model: str,
                              max_retries: int) -> Tuple[List[Image.Image], str]:
        """
        处理单个提示词的图片生成
        
        Args:
            managers: 管理器元组 (config_manager, api_key_manager, task_logger)
            log_id: 任务日志ID
            pil_refs: 参考图片列表
            prompt_text: 提示词
            site_url: 网站URL
            site_name: 网站名称
            model: 模型名称
            max_retries: 最大重试次数
            
        Returns:
            Tuple[List[Image.Image], str]: (生成的图片列表, 错误信息)
        """
        config_manager, api_key_manager, task_logger = managers
        
        for attempt in range(max_retries + 1):
            # 选择API Key
            api_key_config = api_key_manager.get_best_key_for_model(model)
            if not api_key_config:
                error_msg = "没有可用的API Key。请检查配置或等待冷却期结束。"
                if attempt > 0:
                    task_logger.log_retry(log_id, attempt, "没有可用的API Key")
                task_logger.log_error(log_id, error_msg, "api_key")
                return [], error_msg
            
            if attempt > 0:
                task_logger.log_retry(log_id, attempt, f"使用API Key: {api_key_config.get('name', 'Unknown')}")
            
            # 调用API
            pils, error = self._call_openrouter(
                task_logger, log_id, api_key_config, pil_refs, 
                prompt_text, site_url, site_name, model
            )
            
            if not error:
                # 成功，更新API Key统计
                api_key_manager.update_key_stats(api_key_config["id"], True)
                return pils, ""
            
            # 失败，更新API Key统计并标记错误
            api_key_manager.update_key_stats(api_key_config["id"], False)
            api_key_manager.mark_key_error(api_key_config["id"], "error", error)
            
            # 如果这是最后一次尝试，返回错误
            if attempt >= max_retries:
                return [], error
        
        return [], "超出最大重试次数"
    
    def _process_single_prompt_with_key(self, managers: Tuple, log_id: str,
                                       pil_refs: List[Image.Image], prompt_text: str,
                                       site_url: str, site_name: str, model: str,
                                       max_retries: int, assigned_key: Dict[str, Any]) -> Tuple[List[Image.Image], str]:
        """
        使用指定API Key处理单个提示词的图片生成
        
        Args:
            managers: 管理器元组 (config_manager, api_key_manager, task_logger)
            log_id: 任务日志ID
            pil_refs: 参考图片列表
            prompt_text: 提示词
            site_url: 网站URL
            site_name: 网站名称
            model: 模型名称
            max_retries: 最大重试次数
            assigned_key: 指定使用的API Key配置
            
        Returns:
            Tuple[List[Image.Image], str]: (生成的图片列表, 错误信息)
        """
        config_manager, api_key_manager, task_logger = managers
        
        for attempt in range(max_retries + 1):
            # 检查指定Key是否仍然可用
            if not api_key_manager.is_key_available(assigned_key["id"]):
                error_msg = f"指定API Key {assigned_key.get('name', 'Unknown')} 不可用"
                if attempt > 0:
                    task_logger.log_retry(log_id, attempt, error_msg)
                task_logger.log_error(log_id, error_msg, "api_key")
                return [], error_msg
            
            if attempt > 0:
                task_logger.log_retry(log_id, attempt, f"使用API Key: {assigned_key.get('name', 'Unknown')}")
            
            # 调用API
            pils, error = self._call_openrouter(
                task_logger, log_id, assigned_key, pil_refs,
                prompt_text, site_url, site_name, model
            )
            
            if not error:
                # 成功，更新API Key统计
                api_key_manager.update_key_stats(assigned_key["id"], True)
                return pils, ""
            
            # 失败，更新API Key统计并标记错误
            api_key_manager.update_key_stats(assigned_key["id"], False)
            api_key_manager.mark_key_error(assigned_key["id"], "error", error)
            
            # 如果这是最后一次尝试，返回错误
            if attempt >= max_retries:
                return [], error
        
        return [], "超出最大重试次数"
    
    def _process_batch_concurrent(self, managers: Tuple, log_id: str,
                                 prompts: List[str], pil_refs: List[Image.Image],
                                 site_url: str, site_name: str, model: str,
                                 max_concurrent: int, max_retries: int) -> Tuple[List[Image.Image], List[str]]:
        """
        并发处理批量提示词
        
        Args:
            managers: 管理器元组
            log_id: 任务日志ID
            prompts: 提示词列表
            pil_refs: 参考图片列表
            site_url: 网站URL
            site_name: 网站名称
            model: 模型名称
            max_concurrent: 最大并发数
            max_retries: 最大重试次数
            
        Returns:
            Tuple[List[Image.Image], List[str]]: (所有生成的图片, 状态消息列表)
        """
        config_manager, api_key_manager, task_logger = managers
        all_images = []
        status_messages = []
        
        # 获取并发任务所需的API Key
        selected_keys = api_key_manager.select_keys_for_parallel(max_concurrent)
        if not selected_keys:
            return [], ["错误：没有可用的API Key"]
        
        task_logger.log_info(log_id, f"为{max_concurrent}个并发任务选择了{len(selected_keys)}个API Key")
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            try:
                # 提交所有任务
                future_to_info = {}
                for i, prompt_text in enumerate(prompts):
                    if not prompt_text or not prompt_text.strip():
                        status_messages.append(f"第 {i + 1} 行跳过：空提示词")
                        continue
                    
                    # 为每个任务分配一个API Key（轮换使用）
                    assigned_key = selected_keys[i % len(selected_keys)]
                    
                    future = executor.submit(
                        self._process_single_prompt_with_key,
                        managers, log_id, pil_refs, prompt_text.strip(),
                        site_url, site_name, model, max_retries, assigned_key
                    )
                    future_to_info[future] = {
                        "index": i,
                        "key_name": assigned_key.get("name", "Unknown"),
                        "key_id": assigned_key.get("id", "Unknown")
                    }
                
                # 收集结果（修复超时机制和图片保存逻辑）
                completed_futures = set()
                try:
                    # 使用更长的总超时时间，但分批处理
                    for future in as_completed(future_to_info, timeout=60):  # 增加到60秒总超时
                        completed_futures.add(future)
                        info = future_to_info[future]
                        index = info["index"]
                        key_name = info["key_name"]

                        try:
                            pils, error = future.result(timeout=30)  # 增加单个任务超时到30秒
                            if error:
                                status_messages.append(f"图片 {index + 1} 生成失败（使用Key: {key_name}）：{error}")
                            else:
                                all_images.extend(pils)
                                # 计算实际显示的图片数量
                                actual_display_count = get_actual_display_count(pils)

                                # 修复：始终保存生成的图片到输出目录
                                saved_paths = []
                                try:
                                    # 保存所有生成的图片（不仅仅是尺寸不匹配的）
                                    saved_paths = save_images_to_output(
                                        pils,
                                        task_id=log_id or f"concurrent_{index+1}",
                                        prompt=f"并发任务{index+1}"
                                    )

                                    # 记录保存路径到日志
                                    if saved_paths and log_id:
                                        config_manager, api_key_manager, task_logger = managers
                                        if task_logger:
                                            task_logger.log_info(
                                                log_id,
                                                f"图片已保存: {', '.join(saved_paths)}"
                                            )
                                except Exception as save_error:
                                    print(f"[WARNING] 保存图片失败: {save_error}")
                                    status_messages.append(f"图片 {index + 1} 保存失败: {save_error}")

                                # 生成状态消息
                                if len(pils) == actual_display_count:
                                    save_info = f"，已保存到输出目录" if saved_paths else ""
                                    status_messages.append(f"图片 {index + 1} 生成成功（使用Key: {key_name}，{len(pils)} 张{save_info}）。")
                                else:
                                    # 提供保存信息
                                    save_info = f"，已保存到输出目录" if saved_paths else ""
                                    status_messages.append(
                                        f"图片 {index + 1} 生成成功（使用Key: {key_name}，"
                                        f"生成{len(pils)}张，显示{actual_display_count}张{save_info}）。"
                                    )
                        except Exception as e:
                            error_msg = f"图片 {index + 1} 处理异常（使用Key: {key_name}）：{e}"
                            status_messages.append(error_msg)
                            print(f"[ERROR] {error_msg}")

                except Exception as timeout_error:
                    # 处理超时异常，检查未完成的futures
                    unfinished_count = len(future_to_info) - len(completed_futures)
                    if unfinished_count > 0:
                        timeout_msg = f"并发处理超时: {unfinished_count} (of {len(future_to_info)}) futures unfinished"
                        status_messages.append(timeout_msg)
                        print(f"[WARNING] {timeout_msg}")

                        # 调试并发状态
                        self._debug_concurrent_status(future_to_info, completed_futures, log_id)

                        # 尝试获取未完成任务的结果（非阻塞）
                        for future in future_to_info:
                            if future not in completed_futures:
                                info = future_to_info[future]
                                try:
                                    # 非阻塞检查
                                    if future.done():
                                        pils, error = future.result(timeout=0.1)
                                        if not error and pils:
                                            all_images.extend(pils)
                                            # 保存这些图片
                                            try:
                                                saved_paths = save_images_to_output(
                                                    pils,
                                                    task_id=log_id or f"timeout_recovered_{info['index']+1}",
                                                    prompt=f"超时恢复任务{info['index']+1}"
                                                )
                                                if saved_paths:
                                                    status_messages.append(f"图片 {info['index'] + 1} 超时后恢复成功，已保存")
                                            except:
                                                pass
                                except:
                                    pass
                        
            except Exception as e:
                error_msg = f"并发处理异常: {e}"
                status_messages.append(error_msg)
                print(f"[ERROR] {error_msg}")
            finally:
                # 改进的线程池清理逻辑
                try:
                    # 统计未完成的任务
                    unfinished_futures = []
                    for future in future_to_info:
                        if not future.done():
                            unfinished_futures.append(future)

                    if unfinished_futures:
                        print(f"[INFO] 正在清理 {len(unfinished_futures)} 个未完成的任务...")

                        # 尝试取消未完成的任务
                        cancelled_count = 0
                        for future in unfinished_futures:
                            if future.cancel():
                                cancelled_count += 1

                        print(f"[INFO] 成功取消 {cancelled_count}/{len(unfinished_futures)} 个任务")

                    # 优雅关闭线程池
                    executor.shutdown(wait=False)

                except Exception as shutdown_error:
                    print(f"[WARNING] 线程池清理异常: {shutdown_error}")
        
        return all_images, status_messages

    def _process_concurrent_prompts(self, managers: Tuple, log_id: str,
                                  prompts: List[str], pil_refs: List[Image.Image],
                                  site_url: str, site_name: str, model: str,
                                  max_concurrent: int, max_retries: int) -> Tuple[List[Image.Image], List[str]]:
        """
        并发处理多个提示词（主要用于单条prompt的并发生成）
        
        Args:
            managers: 管理器元组
            log_id: 任务日志ID
            prompts: 提示词列表
            pil_refs: 参考图片列表
            site_url: 网站URL
            site_name: 网站名称
            model: 模型名称
            max_concurrent: 最大并发数
            max_retries: 最大重试次数
            
        Returns:
            Tuple[List[Image.Image], List[str]]: (所有生成的图片, 状态消息列表)
        """
        # 复用已有的批量并发处理逻辑
        return self._process_batch_concurrent(
            managers, log_id, prompts, pil_refs,
            site_url, site_name, model, max_concurrent, max_retries
        )
    
    def _debug_concurrent_status(self, future_to_info: dict, completed_futures: set, log_id: str = None):
        """
        调试并发处理状态

        Args:
            future_to_info: Future到信息的映射
            completed_futures: 已完成的Future集合
            log_id: 日志ID
        """
        try:
            total_futures = len(future_to_info)
            completed_count = len(completed_futures)
            pending_count = total_futures - completed_count

            debug_msg = f"并发状态: 总任务={total_futures}, 已完成={completed_count}, 待完成={pending_count}"
            print(f"[DEBUG] {debug_msg}")

            if log_id:
                managers = self._get_managers()
                if managers and managers[2]:  # task_logger
                    task_logger = managers[2]
                    task_logger.log_info(log_id, debug_msg)

            # 详细状态
            for future, info in future_to_info.items():
                status = "completed" if future in completed_futures else ("done" if future.done() else "pending")
                print(f"[DEBUG] Task {info['index']+1} ({info['key_name']}): {status}")

        except Exception as e:
            print(f"[WARNING] 调试状态检查失败: {e}")

    def _cleanup_resources(self, pil_images=None):
        """
        清理资源，防止内存泄漏和UI假死

        Args:
            pil_images: PIL图像列表，可选
        """
        try:
            # 清理PIL图像对象
            if pil_images:
                for img in pil_images:
                    if hasattr(img, 'close'):
                        try:
                            img.close()
                        except:
                            pass

            # 强制垃圾回收
            import gc
            gc.collect()

            # 强制保存待存日志
            try:
                managers = self._get_managers()
                if managers and managers[2]:  # task_logger
                    task_logger = managers[2]
                    if hasattr(task_logger, '_save_pending') and task_logger._save_pending:
                        task_logger._save_task_logs(force=True)
            except:
                pass

        except Exception as e:
            print(f"[WARNING] 资源清理异常: {e}")

    def generate(
        self,
        api_key_main: str,
        image1=None,
        prompt: str = "",
        file_path: str = "",
        site_url: str = "",
        site_name: str = "",
        model: str = "google/gemini-2.5-flash-image-preview:free",
        max_concurrent: int = 3,
        scheduling_mode: str = "round_robin",
        enable_parallel: bool = False,
        max_retries: int = 3,
        enable_detailed_logs: bool = True,
        key_management_mode: str = "同时使用两者",
        auto_refresh_status: bool = True,
        image2=None,
        image3=None,
        image4=None,
        **kwargs  # 捕获所有额外的api_key_X参数
    ):
        """
        新版本的图片生成方法，支持并发、多Key管理和详细日志
        """
        # 获取管理器实例
        try:
            managers = self._get_managers()
            if managers is None or None in managers:
                error_msg = "管理器初始化返回None"
                print(f"警告: {error_msg}")
                return (pils_to_tensor([]), error_msg, "管理器初始化失败")
                
            config_manager, api_key_manager, task_logger = managers
        except Exception as e:
            error_msg = f"初始化管理器失败: {e}"
            print(f"警告: {error_msg}")
            import traceback
            traceback.print_exc()
            # 在管理器初始化失败时返回错误，避免后续使用None对象
            return (pils_to_tensor([]), error_msg, "管理器初始化失败")
        
        # 验证管理器是否正确初始化
        if not all([config_manager, api_key_manager, task_logger]):
            error_msg = "管理器初始化不完整"
            print(f"警告: {error_msg} - config:{config_manager}, api:{api_key_manager}, logger:{task_logger}")
            return (pils_to_tensor([]), error_msg, "管理器初始化不完整")

        # 保存用户选择的key_management_mode到配置文件（如果与当前配置不同）
        current_mode = config_manager.get_setting("key_management_mode", "同时使用两者")
        if key_management_mode != current_mode:
            try:
                config_manager.update_setting("key_management_mode", key_management_mode)
                print(f"[INFO] Key管理模式已更新为: {key_management_mode}")
            except Exception as e:
                print(f"[WARNING] 保存Key管理模式失败: {e}")
        
        # 创建任务ID
        task_id = str(uuid.uuid4())[:8]
        
        # 动态更新API Key状态显示信息（在Key管理模式处理之后）
        current_key_status = ""
        
        # 验证和转换输入图像
        try:
            all_input_pils = validate_and_convert_images([image1, image2, image3, image4])
        except ValueError as e:
            return (pils_to_tensor([]), str(e), "图像验证失败")
        
        # 创建任务日志（如果启用）
        log_id = None
        if enable_detailed_logs:
            task_data = {
                "prompt": prompt,
                "file_path": file_path,
                "model": model,
                "max_concurrent": max_concurrent,
                "max_retries": max_retries,
                "input_images": len(all_input_pils)
            }
            
            if prompt and file_path:
                task_type = "mixed"
            elif file_path:
                task_type = "batch"
            else:
                task_type = "single"
            
            log_id = task_logger.create_task_log(task_id, task_type, task_data)
            task_logger.update_task_status(log_id, TaskStatus.RUNNING)
        
        # 处理多个API Key输入
        input_api_keys = []
        
        # 添加主Key
        if api_key_main and api_key_main.strip():
            input_api_keys.append({
                "name": "Main Key",
                "value": api_key_main.strip(),
                "source": "input"
            })
        
        # 添加额外的Key（从**kwargs中获取）
        for key, value in kwargs.items():
            if key.startswith('api_key_') and value and value.strip():
                key_num = key.replace('api_key_', '')
                input_api_keys.append({
                    "name": f"Key {key_num}",
                    "value": value.strip(),
                    "source": "input"
                })
        
        # 记录临时添加的Key ID
        temp_key_ids = []

        # 根据管理模式处理API Keys
        if key_management_mode == "使用输入的Key":
            # 仅使用输入的Key
            if not input_api_keys:
                return (pils_to_tensor([]), "错误：请至少输入一个API Key", "无API Key")

            print(f"[INFO] 使用输入的Key模式，将使用 {len(input_api_keys)} 个输入Key（仅内存存储）")

            # 设置临时Key模式
            config_manager.set_temp_key_mode(key_management_mode)

            # 添加输入的Key作为临时Key（仅内存，不保存到文件）
            for key_info in input_api_keys:
                key_id = config_manager.add_temp_key(key_info["name"], key_info["value"])
                temp_key_ids.append(key_id)

            # 设置ApiKeyManager的上下文，确保只显示输入Key的状态
            api_key_manager.set_key_context(key_management_mode, temp_key_ids)

        elif key_management_mode == "使用配置文件Key":
            # 仅使用配置文件中的Key，忽略输入的Key
            config_manager.set_temp_key_mode(key_management_mode)
            existing_keys = config_manager.get_api_keys()
            if not existing_keys:
                return (pils_to_tensor([]), "错误：配置文件中没有可用的API Key", "配置中无Key")
            print(f"[INFO] 使用配置文件Key模式，忽略 {len(input_api_keys)} 个输入Key")

            # 设置ApiKeyManager的上下文，确保只显示配置文件Key的状态
            api_key_manager.set_key_context(key_management_mode, [])

        else:  # "同时使用两者"
            # 同时使用输入和配置文件中的Key
            config_manager.set_temp_key_mode(key_management_mode)

            # 添加输入Key作为临时Key（不保存到配置文件）
            for key_info in input_api_keys:
                key_id = config_manager.add_temp_key(key_info["name"], key_info["value"])
                temp_key_ids.append(key_id)

            # 检查是否有可用的Key
            all_keys = config_manager.get_api_keys()
            if not all_keys:
                return (pils_to_tensor([]), "错误：请输入API Key或在配置文件中配置", "无可用Key")

            # 设置ApiKeyManager的上下文，显示所有Key的状态
            api_key_manager.set_key_context(key_management_mode, temp_key_ids)

        # 在Key管理模式处理完成后，刷新当前可用Key的状态
        if auto_refresh_status:
            try:
                current_keys = config_manager.get_api_keys()
                print(f"[INFO] 正在刷新当前 {len(current_keys)} 个API Key的状态...")

                # 只刷新当前配置中的Key（已经根据管理模式过滤）
                refresh_results = self.refresh_key_status()
                print(f"[INFO] Key状态刷新完成: {refresh_results}")

                # 获取状态信息
                current_key_status = self.get_key_status_info(key_management_mode)
            except Exception as e:
                current_key_status = f"获取状态信息失败: {e}"
                print(f"[WARNING] Key状态刷新失败: {e}")

        try:
            all_out_pils = []
            status_msgs = []
            
            # 单条 prompt 处理（支持并发生成多张图片）
            if prompt:
                if max_concurrent > 1:
                    # 并发模式：生成多张图片
                    prompts = [prompt] * max_concurrent  # 复制prompt以支持并发
                    batch_pils, batch_msgs = self._process_concurrent_prompts(
                        managers, log_id or "", prompts, all_input_pils,
                        site_url, site_name, model, max_concurrent, max_retries
                    )
                    all_out_pils.extend(batch_pils)
                    status_msgs.extend(batch_msgs)
                    if not batch_pils:
                        error_msg = "并发生成失败，未生成任何图片"
                        if log_id:
                            task_logger.update_task_status(log_id, TaskStatus.FAILED, error_msg)
                        return (pils_to_tensor(all_input_pils), error_msg, "并发处理失败")
                else:
                    # 单张模式：仅生成一张图片
                    pils, error = self._process_single_prompt(
                        managers, log_id or "", all_input_pils, prompt,
                        site_url, site_name, model, max_retries
                    )
                    if error:
                        if log_id:
                            task_logger.update_task_status(log_id, TaskStatus.FAILED, error)
                        return (pils_to_tensor(all_input_pils), error, "单图生成失败")
                    
                    all_out_pils.extend(pils)
                    # 计算实际显示的图片数量
                    actual_display_count = get_actual_display_count(pils)

                    # 修复：始终保存生成的图片到输出目录
                    saved_paths = []
                    try:
                        saved_paths = save_images_to_output(
                            pils,
                            task_id=log_id or "single_prompt",
                            prompt=prompt[:50] if prompt else "单图生成"  # 限制提示词长度
                        )

                        # 记录保存路径到日志
                        if saved_paths and log_id:
                            task_logger.log_info(
                                log_id,
                                f"图片已保存: {', '.join(saved_paths)}"
                            )
                    except Exception as save_error:
                        print(f"[WARNING] 保存图片失败: {save_error}")
                        status_msgs.append(f"图片保存失败: {save_error}")

                    # 生成状态消息
                    save_info = f"，已保存到输出目录" if saved_paths else ""
                    if len(pils) == actual_display_count:
                        status_msgs.append(f"已生成 {len(pils)} 张图片{save_info}。")
                    else:
                        status_msgs.append(f"已生成 {len(pils)} 张图片，显示 {actual_display_count} 张{save_info}。")
            
            # 批量文件处理 
            elif file_path:
                batch_pils, batch_msgs = self._process_batch_file(
                    managers, log_id or "", all_input_pils, file_path,
                    site_url, site_name, model, max_concurrent, max_retries
                )
                all_out_pils.extend(batch_pils)
                status_msgs.extend(batch_msgs)
            
            if not all_out_pils:
                error_msg = "未生成任何图片。"
                if log_id:
                    task_logger.update_task_status(log_id, TaskStatus.FAILED, error_msg)
                return (pils_to_tensor(all_input_pils), error_msg + "\n" + "\n".join(status_msgs), "生成失败")
            
            # 成功完成
            if log_id:
                task_logger.update_task_status(log_id, TaskStatus.SUCCESS)
                # 强制保存日志，确保任务完成状态被记录
                task_logger._save_task_logs(force=True)
            
            # 清理临时添加的Key（如果需要）
            if key_management_mode in ["使用输入的Key", "同时使用两者"] and input_api_keys:
                # 仅在使用输入Key时清理临时Key
                cleaned_count = config_manager.cleanup_temporary_keys()
                if cleaned_count > 0:
                    status_msgs.append(f"已清理 {cleaned_count} 个临时API Key")
            
            # 转换输出结果
            out_tensor = pils_to_tensor(all_out_pils)
            
            # 检查尺寸不匹配问题
            size_info = create_size_mismatch_message(all_out_pils)
            
            # 组合状态信息（包含API Key状态）
            final_status = ("\n".join(status_msgs) + size_info) if status_msgs else ("完成" + size_info)
            
            # 获取实时Key状态信息
            final_key_status = self.get_key_status_info(key_management_mode) if auto_refresh_status else current_key_status
            
            # 清理资源，防止内存泄漏
            self._cleanup_resources(all_out_pils)

            # 清理临时Key和上下文
            config_manager.clear_temp_keys()

            # 清除ApiKeyManager的上下文
            api_key_manager.clear_key_context()

            return (out_tensor, final_status, final_key_status)
            
        except Exception as e:
            error_msg = f"任务执行异常: {e}"
            if log_id:
                task_logger.log_error(log_id, error_msg, "execution")
                task_logger.update_task_status(log_id, TaskStatus.FAILED, error_msg)
            
            # 在异常情况下也要清理资源
            self._cleanup_resources(all_input_pils)

            # 清理临时Key和上下文
            config_manager.clear_temp_keys()

            # 清除ApiKeyManager的上下文
            api_key_manager.clear_key_context()

            return (pils_to_tensor(all_input_pils), error_msg, "执行异常")
    
    def _process_batch_file(self, managers: Tuple, log_id: str,
                           pil_refs: List[Image.Image], file_path: str,
                           site_url: str, site_name: str, model: str,
                           max_concurrent: int, max_retries: int) -> Tuple[List[Image.Image], List[str]]:
        """处理批量文件"""
        config_manager, api_key_manager, task_logger = managers
        
        # 改进的路径处理
        clean_path = file_path.strip()
        if (clean_path.startswith('"') and clean_path.endswith('"')) or \
           (clean_path.startswith("'") and clean_path.endswith("'")):
            clean_path = clean_path[1:-1]
        
        import re
        clean_path = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_path)
        clean_path = os.path.normpath(clean_path)
        
        if not os.path.exists(clean_path):
            return [], [f"错误：文件路径不存在: {clean_path}"]
        
        if not HAS_PANDAS:
            return [], ["错误：批量模式需要 pandas"]
        
        try:
            if clean_path.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(clean_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(clean_path, encoding='gbk')
                    except UnicodeDecodeError:
                        df = pd.read_csv(clean_path, encoding='latin1')
            else:
                df = pd.read_excel(clean_path, sheet_name="Sheet1")
        except Exception as e:
            return [], [f"读取文件失败：{e}"]
        
        if "prompt" not in df.columns:
            return [], ["错误：文件中未找到 'prompt' 列"]
        
        prompts = [row.get("prompt", "") for _, row in df.iterrows()]
        return self._process_batch_concurrent(
            managers, log_id, prompts, pil_refs,
            site_url, site_name, model, max_concurrent, max_retries
        )


# 注册到 ComfyUI
# 2. 修改节点类映射，使用新的类名作为键和值
NODE_CLASS_MAPPINGS = {
    "GoogleNanoNode": GoogleNanoNode,
}
# 3. 修改节点显示名称映射，使用新的类名作为键
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleNanoNode": "google nano",
}
