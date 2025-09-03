"""
API Key管理器 - 负责API Key的状态监控、调度策略和负载均衡
"""

import random
import threading
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from .config_manager import ConfigManager


class ApiKeyStatus:
    """API Key状态枚举"""
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID = "invalid"
    ERROR = "error"
    COOLDOWN = "cooldown"


class ApiKeyManager:
    """API Key管理器"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化API Key管理器

        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self._lock = threading.RLock()
        # 添加上下文管理
        self._current_filter_mode = None
        self._current_temp_key_ids = None

    def set_key_context(self, filter_mode: str = None, temp_key_ids: List[str] = None):
        """
        设置Key过滤上下文

        Args:
            filter_mode: 过滤模式
            temp_key_ids: 临时Key ID列表
        """
        with self._lock:
            self._current_filter_mode = filter_mode
            self._current_temp_key_ids = temp_key_ids

    def clear_key_context(self):
        """清除Key过滤上下文"""
        with self._lock:
            self._current_filter_mode = None
            self._current_temp_key_ids = None
        self._current_key_index = 0
        self._key_usage_stats = {}
        self._last_status_check = 0
        
        # 状态检查间隔（秒）
        self._status_check_interval = 300  # 5分钟
        
        # 冷却时间设置（分钟）
        self._cooldown_times = {
            ApiKeyStatus.RATE_LIMITED: 10,  # 速率限制冷却10分钟
            ApiKeyStatus.QUOTA_EXCEEDED: 60,  # 配额超限冷却1小时
            ApiKeyStatus.ERROR: 5  # 一般错误冷却5分钟
        }
    
    def is_key_available(self, key_id: str) -> bool:
        """
        检查指定API Key是否可用
        
        Args:
            key_id: API Key ID
            
        Returns:
            是否可用
        """
        available_keys = self.get_available_keys()
        return any(key.get("id") == key_id for key in available_keys)
    
    def get_available_keys(self) -> List[Dict[str, Any]]:
        """获取所有可用的API Key（优化了锁的使用）"""
        # 先获取配置，使用当前上下文过滤
        with self._lock:
            filter_mode = self._current_filter_mode
            temp_key_ids = self._current_temp_key_ids

        api_keys = self.config_manager.get_api_keys(filter_mode, temp_key_ids)
        available_keys = []
        current_time = datetime.now()
        
        # 在锁外处理大部分逻辑
        keys_to_reset = []
        
        for key_config in api_keys:
            # 检查冷却时间
            cooldown_until = key_config.get("cooldown_until")
            if cooldown_until:
                try:
                    cooldown_time = datetime.fromisoformat(cooldown_until)
                    if current_time < cooldown_time:
                        continue  # 仍在冷却期，跳过此Key
                    else:
                        # 冷却期结束，记录需要重置的Key
                        keys_to_reset.append(key_config["id"])
                except (ValueError, TypeError):
                    # 如果cooldown_until格式不正确，记录需要重置的Key
                    keys_to_reset.append(key_config["id"])
            
            # 只返回可用状态的Key
            if key_config.get("status") == ApiKeyStatus.AVAILABLE:
                available_keys.append(key_config.copy())
        
        # 仅在需要时使用锁来重置状态
        if keys_to_reset:
            with self._lock:
                for key_id in keys_to_reset:
                    self._reset_key_status(key_id)
        
        return available_keys
    
    def _reset_key_status(self, key_id: str):
        """重置API Key状态为可用"""
        self.config_manager.update_api_key_status(
            key_id, 
            ApiKeyStatus.AVAILABLE,
            cooldown_until=None
        )
    
    def select_key_round_robin(self) -> Optional[Dict[str, Any]]:
        """轮询模式选择API Key"""
        with self._lock:
            available_keys = self.get_available_keys()
            
            if not available_keys:
                return None
            
            # 使用轮询算法
            key = available_keys[self._current_key_index % len(available_keys)]
            self._current_key_index = (self._current_key_index + 1) % len(available_keys)
            
            return key
    
    def select_key_random(self) -> Optional[Dict[str, Any]]:
        """随机模式选择API Key"""
        available_keys = self.get_available_keys()
        
        if not available_keys:
            return None
        
        return random.choice(available_keys)
    
    def select_key_weighted(self) -> Optional[Dict[str, Any]]:
        """加权模式选择API Key（基于剩余配额和成功率）"""
        available_keys = self.get_available_keys()
        
        if not available_keys:
            return None
        
        # 计算每个Key的权重
        weights = []
        for key_config in available_keys:
            stats = key_config.get("stats", {})
            
            # 基础权重：剩余配额
            remaining_quota = stats.get("daily_remaining", 1000)
            base_weight = max(remaining_quota, 1)  # 至少为1
            
            # 成功率加权
            success_count = stats.get("success_count", 0)
            error_count = stats.get("error_count", 0)
            total_requests = success_count + error_count
            
            if total_requests > 0:
                success_rate = success_count / total_requests
                success_weight = success_rate * 2  # 成功率影响权重
            else:
                success_weight = 1.0  # 新Key默认权重
            
            final_weight = base_weight * success_weight
            weights.append(final_weight)
        
        # 根据权重随机选择
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(available_keys)
        
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for i, weight in enumerate(weights):
            current_weight += weight
            if random_value <= current_weight:
                return available_keys[i]
        
        # 备用：返回最后一个
        return available_keys[-1]
    
    def select_key(self, mode: str = None) -> Optional[Dict[str, Any]]:
        """
        根据指定模式选择API Key
        
        Args:
            mode: 选择模式，如果为None则使用配置中的默认模式
            
        Returns:
            选中的API Key配置，如果没有可用Key则返回None
        """
        if mode is None:
            mode = self.config_manager.get_setting("scheduling_mode", "round_robin")
        
        if mode == "random":
            return self.select_key_random()
        elif mode == "weighted":
            return self.select_key_weighted()
        else:  # 默认为round_robin
            return self.select_key_round_robin()
    
    def select_keys_for_parallel(self, count: int) -> List[Dict[str, Any]]:
        """
        为并行处理选择多个API Key
        
        Args:
            count: 需要的Key数量
            
        Returns:
            选中的API Key列表
        """
        available_keys = self.get_available_keys()
        
        if not available_keys:
            return []
        
        # 如果可用Key数量少于需求，使用全部Key
        if len(available_keys) <= count:
            return available_keys
        
        # 根据加权算法选择多个Key，避免重复
        selected_keys = []
        keys_pool = available_keys.copy()
        
        for _ in range(count):
            if not keys_pool:
                break
            
            # 使用加权选择
            selected_key = self._select_weighted_from_pool(keys_pool)
            selected_keys.append(selected_key)
            
            # 从池中移除已选择的Key
            keys_pool = [k for k in keys_pool if k["id"] != selected_key["id"]]
        
        return selected_keys
    
    def _select_weighted_from_pool(self, keys_pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从Key池中基于权重选择一个Key"""
        if len(keys_pool) == 1:
            return keys_pool[0]
        
        # 计算权重（类似于select_key_weighted方法）
        weights = []
        for key_config in keys_pool:
            stats = key_config.get("stats", {})
            remaining_quota = stats.get("daily_remaining", 1000)
            base_weight = max(remaining_quota, 1)
            
            success_count = stats.get("success_count", 0)
            error_count = stats.get("error_count", 0)
            total_requests = success_count + error_count
            
            if total_requests > 0:
                success_rate = success_count / total_requests
                success_weight = success_rate * 2
            else:
                success_weight = 1.0
            
            final_weight = base_weight * success_weight
            weights.append(final_weight)
        
        # 根据权重选择
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(keys_pool)
        
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for i, weight in enumerate(weights):
            current_weight += weight
            if random_value <= current_weight:
                return keys_pool[i]
        
        return keys_pool[-1]
    
    def update_key_stats(self, key_id: str, success: bool, response_data: Dict[str, Any] = None):
        """
        更新API Key的使用统计
        
        Args:
            key_id: API Key ID
            success: 请求是否成功
            response_data: API响应数据，用于解析限制信息
        """
        with self._lock:
            api_keys = self.config_manager.get_api_keys()
            
            for key_config in api_keys:
                if key_config.get("id") == key_id:
                    stats = key_config.setdefault("stats", {})
                    
                    # 更新计数器
                    if success:
                        stats["success_count"] = stats.get("success_count", 0) + 1
                    else:
                        stats["error_count"] = stats.get("error_count", 0) + 1
                    
                    stats["total_requests"] = stats.get("total_requests", 0) + 1
                    stats["last_used"] = datetime.now().isoformat()
                    
                    # 从响应中解析限制信息
                    if response_data:
                        self._parse_rate_limit_info(key_id, response_data)
                    
                    # 更新配置
                    self.config_manager.update_api_key_status(
                        key_id, 
                        key_config.get("status", ApiKeyStatus.AVAILABLE),
                        stats
                    )
                    break
    
    def _parse_rate_limit_info(self, key_id: str, response_data: Dict[str, Any]):
        """从API响应中解析速率限制信息"""
        try:
            # 这里可以根据OpenRouter API的实际响应格式来解析
            # 目前作为占位符实现
            headers = response_data.get("headers", {})
            
            # 解析剩余配额信息（如果API返回）
            remaining_quota = headers.get("x-ratelimit-remaining")
            if remaining_quota:
                try:
                    remaining = int(remaining_quota)
                    self.config_manager.update_api_key_status(
                        key_id,
                        ApiKeyStatus.AVAILABLE,
                        {"daily_remaining": remaining}
                    )
                except (ValueError, TypeError):
                    pass
            
        except Exception as e:
            print(f"解析速率限制信息失败: {e}")
    
    def mark_key_error(self, key_id: str, error_type: str, error_message: str = ""):
        """
        标记API Key出现错误
        
        Args:
            key_id: API Key ID
            error_type: 错误类型（rate_limit, quota_exceeded, invalid, error）
            error_message: 错误详情
        """
        with self._lock:
            # 确定新状态
            if "rate" in error_message.lower() or "429" in error_message:
                new_status = ApiKeyStatus.RATE_LIMITED
            elif "quota" in error_message.lower() or "limit" in error_message.lower():
                new_status = ApiKeyStatus.QUOTA_EXCEEDED
            elif "auth" in error_message.lower() or "401" in error_message or "403" in error_message:
                new_status = ApiKeyStatus.INVALID
            else:
                new_status = ApiKeyStatus.ERROR
            
            # 计算冷却结束时间
            cooldown_minutes = self._cooldown_times.get(new_status, 5)
            cooldown_until = (datetime.now() + timedelta(minutes=cooldown_minutes)).isoformat()
            
            # 更新错误统计
            stats = {"error_count": 1, "last_error": error_message}
            
            # 更新Key状态
            self.config_manager.update_api_key_status(
                key_id,
                new_status,
                stats,
                cooldown_until
            )
            
            print(f"API Key {key_id} 标记为 {new_status}，冷却时间至 {cooldown_until}")
    
    def check_key_status(self, key_id: str, api_key_value: str) -> bool:
        """
        检查API Key的当前状态（通过实际API调用）
        
        Args:
            key_id: API Key ID
            api_key_value: API Key值
            
        Returns:
            Key是否可用
        """
        try:
            # 调用OpenRouter官方API检查Key状态
            key_info = self._fetch_key_info_from_openrouter(api_key_value)
            if key_info:
                # 更新Key状态信息
                self._update_key_from_openrouter_response(key_id, key_info)
                return True
            else:
                self.mark_key_error(key_id, "invalid", "无法获取Key状态信息")
                return False
        except Exception as e:
            self.mark_key_error(key_id, "error", str(e))
            return False
    
    def _fetch_key_info_from_openrouter(self, api_key_value: str) -> Optional[Dict[str, Any]]:
        """
        从 OpenRouter API 获取 Key 信息
        
        Args:
            api_key_value: API Key值
            
        Returns:
            Key信息字典或None
        """
        try:
            url = "https://openrouter.ai/api/v1/key"
            headers = {
                "Authorization": f"Bearer {api_key_value}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("data")
            elif response.status_code == 401:
                print(f"API Key 无效: {api_key_value[:10]}...")
                return None
            elif response.status_code == 402:
                print(f"API Key 余额不足: {api_key_value[:10]}...")
                return None
            else:
                print(f"OpenRouter API 调用失败: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("OpenRouter API 调用超时")
            return None
        except Exception as e:
            print(f"OpenRouter API 调用异常: {e}")
            return None
    
    def _update_key_from_openrouter_response(self, key_id: str, key_info: Dict[str, Any]):
        """
        根据 OpenRouter API 响应更新 Key 状态
        
        Args:
            key_id: API Key ID
            key_info: OpenRouter API返回的Key信息
        """
        try:
            # 解析OpenRouter API响应
            label = key_info.get("label", "")
            usage = key_info.get("usage", 0)  # 已使用的积分
            limit = key_info.get("limit")  # 积分限制，null表示无限制
            is_free_tier = key_info.get("is_free_tier", False)
            
            # 计算剩余积分
            if limit is not None:
                remaining = max(0, limit - usage)
            else:
                remaining = "unlimited"
            
            # 决定 Key 状态
            if limit is not None and usage >= limit:
                status = ApiKeyStatus.QUOTA_EXCEEDED
            elif usage < 0:  # 负余额
                status = ApiKeyStatus.QUOTA_EXCEEDED
            else:
                status = ApiKeyStatus.AVAILABLE
            
            # 准备统计信息
            stats = {
                "usage": usage,
                "limit": limit,
                "remaining": remaining,
                "is_free_tier": is_free_tier,
                "label": label,
                "last_checked": datetime.now().isoformat()
            }
            
            # 更新配置
            self.config_manager.update_api_key_status(
                key_id,
                status,
                stats,
                None  # 清除冷却时间
            )
            
            print(f"Key {key_id} 状态更新: {status}, 余额: {remaining}")
            
        except Exception as e:
            print(f"更新Key状态失败: {e}")
            
    def refresh_all_keys_status(self) -> Dict[str, bool]:
        """
        刷新所有API Key的状态

        Returns:
            各Key的状态检查结果
        """
        results = {}
        # 使用当前上下文获取需要刷新的Key
        with self._lock:
            filter_mode = self._current_filter_mode
            temp_key_ids = self._current_temp_key_ids

        api_keys = self.config_manager.get_api_keys(filter_mode, temp_key_ids)
        
        print(f"[INFO] 开始刷新 {len(api_keys)} 个API Key的状态...")
        
        for key_config in api_keys:
            key_id = key_config.get("id")
            key_value = key_config.get("value")
            key_name = key_config.get("name", key_id)
            
            if key_id and key_value:
                print(f"[INFO] 检查Key: {key_name}")
                success = self.check_key_status(key_id, key_value)
                results[key_id] = success
            else:
                print(f"[WARNING] Key配置不完整: {key_name}")
                results[key_id] = False
        
        print(f"[INFO] Key状态刷新完成: {sum(results.values())}/{len(results)} 个可用")
        return results
    
    def get_key_statistics(self) -> Dict[str, Any]:
        """获取所有Key的统计信息（包含真实OpenRouter状态）"""
        # 使用当前上下文获取Key统计
        with self._lock:
            filter_mode = self._current_filter_mode
            temp_key_ids = self._current_temp_key_ids

        api_keys = self.config_manager.get_api_keys(filter_mode, temp_key_ids)
        available_keys = self.get_available_keys()
        
        stats = {
            "total_keys": len(api_keys),
            "available_keys": len(available_keys),
            "key_details": []
        }
        
        for key_config in api_keys:
            key_stats = key_config.get("stats", {})
            
            # 从 OpenRouter API 响应中获取真实数据
            usage = key_stats.get("usage", 0)
            limit = key_stats.get("limit")
            remaining = key_stats.get("remaining", "未知")
            is_free_tier = key_stats.get("is_free_tier", False)
            label = key_stats.get("label", "")
            
            detail = {
                "id": key_config.get("id"),
                "name": key_config.get("name"),
                "status": key_config.get("status"),
                "success_rate": 0,
                "usage": usage,
                "limit": limit if limit is not None else "无限制",
                "remaining": remaining,
                "is_free_tier": is_free_tier,
                "label": label,
                "total_requests": key_stats.get("total_requests", 0),
                "last_used": key_stats.get("last_used"),
                "last_checked": key_stats.get("last_checked")
            }
            
            # 计算成功率
            success_count = key_stats.get("success_count", 0)
            error_count = key_stats.get("error_count", 0)
            total_requests = success_count + error_count
            
            if total_requests > 0:
                detail["success_rate"] = round((success_count / total_requests) * 100, 2)
            
            stats["key_details"].append(detail)
        
        return stats
    
    def cleanup_expired_cooldowns(self):
        """清理已过期的冷却状态"""
        with self._lock:
            api_keys = self.config_manager.get_api_keys()
            current_time = datetime.now()
            
            for key_config in api_keys:
                cooldown_until = key_config.get("cooldown_until")
                if cooldown_until:
                    try:
                        cooldown_time = datetime.fromisoformat(cooldown_until)
                        if current_time >= cooldown_time:
                            # 冷却期结束，重置状态
                            self._reset_key_status(key_config["id"])
                    except (ValueError, TypeError):
                        # 如果cooldown_until格式不正确，重置状态
                        self._reset_key_status(key_config["id"])
    
    def get_best_key_for_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        根据模型类型选择最合适的API Key
        
        Args:
            model_name: 模型名称
            
        Returns:
            最适合的API Key配置
        """
        # 检查是否为免费模型
        is_free_model = model_name.endswith(":free")
        
        available_keys = self.get_available_keys()
        if not available_keys:
            return None
        
        # 如果是免费模型，优先选择日配额较多的Key
        if is_free_model:
            # 按日剩余配额排序
            free_suitable_keys = []
            for key in available_keys:
                stats = key.get("stats", {})
                daily_remaining = stats.get("daily_remaining", 1000)
                if daily_remaining > 0:  # 还有免费配额
                    free_suitable_keys.append((key, daily_remaining))
            
            if free_suitable_keys:
                # 选择剩余配额最多的Key
                best_key = max(free_suitable_keys, key=lambda x: x[1])[0]
                return best_key
        
        # 对于付费模型或没有合适免费配额的情况，使用正常选择策略
        return self.select_key()