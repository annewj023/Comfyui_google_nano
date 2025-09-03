"""
配置管理器 - 负责config.json文件的读写、验证和热重载功能
"""

import json
import os
import threading
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid


class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径为当前模块所在目录的config.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(os.path.dirname(current_dir), "config.json")
        else:
            self.config_path = config_path
            
        self._config = {}
        self._lock = threading.RLock()
        self._last_modified = 0
        self._auto_reload = True
        # 添加临时Key存储（仅内存，不保存到文件）
        self._temp_keys = []
        self._temp_key_mode = None
        
        # 确保配置文件存在
        self._ensure_config_exists()
        # 加载配置
        self.reload_config()

    def set_temp_key_mode(self, mode: str):
        """
        设置临时Key模式

        Args:
            mode: Key管理模式
        """
        with self._lock:
            self._temp_key_mode = mode
            if mode != "使用输入的Key":
                # 非输入Key模式时清空临时Key
                self._temp_keys = []

    def add_temp_key(self, name: str, value: str) -> str:
        """
        添加临时Key（仅内存，不保存到文件）

        Args:
            name: Key名称
            value: Key值

        Returns:
            生成的Key ID
        """
        import uuid
        key_id = str(uuid.uuid4())[:8]

        temp_key = {
            "id": key_id,
            "name": name,
            "value": value,
            "status": "available",
            "created_at": datetime.now().isoformat(),
            "stats": {
                "success_count": 0,
                "error_count": 0,
                "daily_remaining": 1000
            }
        }

        with self._lock:
            self._temp_keys.append(temp_key)

        print(f"[INFO] 临时添加输入Key: {name} (ID: {key_id}) - 仅内存存储")
        return key_id

    def clear_temp_keys(self):
        """清空临时Key"""
        with self._lock:
            self._temp_keys = []
            self._temp_key_mode = None
        print("[INFO] 已清空临时Key存储")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "version": "1.0",
            "api_keys": [],
            "settings": {
                "max_concurrent": 5,
                "scheduling_mode": "round_robin",  # round_robin, random, weighted
                "enable_parallel": False,
                "retry_attempts": 3,
                "retry_delay": 1.0,
                "log_level": "INFO",
                "auto_reload": True,
                "key_status_check_interval": 300,  # 5分钟检查一次Key状态
                "encryption_enabled": False,
                "always_save_images": True,
                "concurrent_timeout": 60,
                "individual_task_timeout": 30,
                "key_management_mode": "同时使用两者"  # 使用输入的Key, 使用配置文件Key, 同时使用两者
            },
            "models": [
                "google/gemini-2.5-flash-image-preview:free",
                "google/gemini-2.5-flash-image-preview",
                "google/gemini-1.5-pro:free",
                "google/gemini-1.5-pro"
            ],
            "rate_limits": {
                "free_model_daily_limit": 50,
                "free_model_minute_limit": 20,
                "paid_model_minute_limit": 60
            }
        }
    
    def _ensure_config_exists(self):
        """确保配置文件存在，如果不存在则创建默认配置"""
        if not os.path.exists(self.config_path):
            print(f"配置文件不存在，创建默认配置: {self.config_path}")
            self._save_config(self._get_default_config())
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到: {self.config_path}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            raise
    
    def reload_config(self) -> bool:
        """重新加载配置文件"""
        try:
            with self._lock:
                if os.path.exists(self.config_path):
                    # 检查文件修改时间
                    current_modified = os.path.getmtime(self.config_path)
                    if current_modified <= self._last_modified and self._config:
                        return False  # 文件未修改
                    
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        new_config = json.load(f)
                    
                    # 验证配置格式
                    if self._validate_config(new_config):
                        self._config = new_config
                        self._last_modified = current_modified
                        self._auto_reload = self._config.get("settings", {}).get("auto_reload", True)
                        print(f"配置文件已重新加载: {self.config_path}")
                        return True
                    else:
                        print("配置文件格式验证失败，保持当前配置")
                        return False
                else:
                    print(f"配置文件不存在: {self.config_path}")
                    return False
        except Exception as e:
            print(f"重新加载配置文件失败: {e}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置格式是否正确"""
        try:
            # 检查必需的顶级键
            required_keys = ["version", "api_keys", "settings", "models"]
            for key in required_keys:
                if key not in config:
                    print(f"配置文件缺少必需的键: {key}")
                    return False
            
            # 检查api_keys格式
            if not isinstance(config["api_keys"], list):
                print("api_keys必须是列表")
                return False
            
            # 检查settings格式
            settings = config.get("settings", {})
            if not isinstance(settings, dict):
                print("settings必须是字典")
                return False
            
            # 检查关键设置项
            max_concurrent = settings.get("max_concurrent", 5)
            if not isinstance(max_concurrent, int) or max_concurrent < 1 or max_concurrent > 100:
                print("max_concurrent必须是1-100之间的整数")
                return False
            
            scheduling_mode = settings.get("scheduling_mode", "round_robin")
            if scheduling_mode not in ["round_robin", "random", "weighted"]:
                print("scheduling_mode必须是: round_robin, random, weighted 之一")
                return False
            
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        with self._lock:
            if self._auto_reload:
                self.reload_config()
            return self._config.copy()
    
    def get_setting(self, key: str, default_value: Any = None) -> Any:
        """获取设置项的值"""
        config = self.get_config()
        return config.get("settings", {}).get(key, default_value)
    
    def update_setting(self, key: str, value: Any) -> bool:
        """更新设置项"""
        try:
            with self._lock:
                config = self._config.copy()
                if "settings" not in config:
                    config["settings"] = {}
                config["settings"][key] = value
                
                if self._validate_config(config):
                    self._save_config(config)
                    self._config = config
                    return True
                else:
                    print(f"无效的设置值: {key} = {value}")
                    return False
        except Exception as e:
            print(f"更新设置失败: {e}")
            return False
    
    def find_key_by_value(self, key_value: str) -> Optional[Dict[str, Any]]:
        """
        根据API Key值查找已存在的Key配置
        
        Args:
            key_value: 要查找的API Key值
            
        Returns:
            如果找到则返回Key配置字典，否则返回None
        """
        try:
            with self._lock:
                config = self._config.copy()
                for key_config in config.get("api_keys", []):
                    if key_config.get("value") == key_value:
                        return key_config
                return None
        except Exception as e:
            print(f"查找API Key失败: {e}")
            return None
    
    def get_api_keys(self, filter_mode: str = None, temp_key_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        获取API Key配置

        Args:
            filter_mode: 过滤模式 ("使用输入的Key", "使用配置文件Key", "同时使用两者", None)
            temp_key_ids: 临时添加的Key ID列表（用于"使用输入的Key"模式）

        Returns:
            根据模式过滤后的API Key列表
        """
        config = self.get_config()
        config_keys = config.get("api_keys", [])

        with self._lock:
            temp_keys = self._temp_keys.copy()
            current_temp_mode = self._temp_key_mode

        # 如果没有指定过滤模式，返回配置文件Key + 临时Key（保持向后兼容）
        if filter_mode is None:
            if current_temp_mode == "使用输入的Key":
                # 输入Key模式下只返回临时Key
                return temp_keys
            else:
                # 其他模式返回配置Key + 临时Key
                return config_keys + temp_keys

        if filter_mode == "使用输入的Key":
            # 只返回临时Key
            return temp_keys

        elif filter_mode == "使用配置文件Key":
            # 只返回配置文件Key
            return config_keys

        else:  # "同时使用两者" 或其他情况
            # 返回配置Key + 临时Key
            # 注意：这里需要检查当前的临时Key模式
            if current_temp_mode == "同时使用两者":
                return config_keys + temp_keys
            else:
                # 如果不是"同时使用两者"模式，只返回配置Key
                return config_keys
    
    def add_api_key(self, name: str, value: str, encrypted: bool = False) -> str:
        """
        添加新的API Key
        
        Args:
            name: Key的显示名称
            value: Key的值（可能已加密）
            encrypted: 值是否已加密
            
        Returns:
            新Key的ID，如果已存在相同Key值则返回现有Key的ID
        """
        try:
            with self._lock:
                # 检查是否已存在相同的API Key值
                existing_key = self.find_key_by_value(value)
                if existing_key:
                    print(f"API Key已存在，跳过添加: {existing_key.get('name', 'Unknown')} (ID: {existing_key.get('id', 'Unknown')})")
                    return existing_key.get('id', '')
                
                config = self._config.copy()
                
                # 生成唯一ID
                key_id = str(uuid.uuid4())[:8]
                
                new_key = {
                    "id": key_id,
                    "name": name,
                    "value": value,
                    "encrypted": encrypted,
                    "status": "available",
                    "stats": {
                        "daily_remaining": 1000,
                        "last_used": None,
                        "success_count": 0,
                        "error_count": 0,
                        "total_requests": 0
                    },
                    "cooldown_until": None,
                    "created_at": datetime.now().isoformat()
                }
                
                config["api_keys"].append(new_key)
                
                if self._validate_config(config):
                    self._save_config(config)
                    self._config = config
                    print(f"API Key已添加: {name} (ID: {key_id})")
                    return key_id
                else:
                    print("配置验证失败，无法添加API Key")
                    return ""
        except Exception as e:
            print(f"添加API Key失败: {e}")
            return ""
    
    def remove_api_key(self, key_id: str) -> bool:
        """删除API Key"""
        try:
            with self._lock:
                config = self._config.copy()
                
                # 查找并删除指定的Key
                original_length = len(config["api_keys"])
                config["api_keys"] = [key for key in config["api_keys"] if key.get("id") != key_id]
                
                if len(config["api_keys"]) < original_length:
                    self._save_config(config)
                    self._config = config
                    print(f"API Key已删除: {key_id}")
                    return True
                else:
                    print(f"未找到指定的API Key: {key_id}")
                    return False
        except Exception as e:
            print(f"删除API Key失败: {e}")
            return False
    
    def update_api_key_status(self, key_id: str, status: str, stats: Dict[str, Any] = None, cooldown_until: str = None) -> bool:
        """更新API Key状态"""
        try:
            with self._lock:
                config = self._config.copy()
                
                for key_config in config["api_keys"]:
                    if key_config.get("id") == key_id:
                        key_config["status"] = status
                        if stats:
                            key_config.setdefault("stats", {}).update(stats)
                        if cooldown_until is not None:
                            key_config["cooldown_until"] = cooldown_until
                        
                        self._save_config(config)
                        self._config = config
                        return True
                
                print(f"未找到指定的API Key: {key_id}")
                return False
        except Exception as e:
            print(f"更新API Key状态失败: {e}")
            return False
    
    def get_models(self) -> List[str]:
        """获取可用模型列表"""
        config = self.get_config()
        return config.get("models", [])
    
    def add_model(self, model_name: str) -> bool:
        """添加新模型"""
        try:
            with self._lock:
                config = self._config.copy()
                
                if model_name not in config.get("models", []):
                    config.setdefault("models", []).append(model_name)
                    self._save_config(config)
                    self._config = config
                    print(f"模型已添加: {model_name}")
                    return True
                else:
                    print(f"模型已存在: {model_name}")
                    return False
        except Exception as e:
            print(f"添加模型失败: {e}")
            return False
    
    def get_rate_limits(self) -> Dict[str, int]:
        """获取速率限制配置"""
        config = self.get_config()
        return config.get("rate_limits", {})
    
    def export_config(self, export_path: str = None) -> str:
        """导出配置文件"""
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"config_backup_{timestamp}.json"
        
        try:
            config = self.get_config()
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"配置已导出到: {export_path}")
            return export_path
        except Exception as e:
            print(f"导出配置失败: {e}")
            return ""
    
    def import_config(self, import_path: str) -> bool:
        """导入配置文件"""
        try:
            if not os.path.exists(import_path):
                print(f"导入文件不存在: {import_path}")
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            if self._validate_config(imported_config):
                with self._lock:
                    self._save_config(imported_config)
                    self._config = imported_config
                print(f"配置已从文件导入: {import_path}")
                return True
            else:
                print("导入的配置文件格式验证失败")
                return False
        except Exception as e:
            print(f"导入配置失败: {e}")
            return False
    
    def cleanup_temporary_keys(self) -> int:
        """
        清理临时添加的API Key（名称包含temp或Main Key的）
        
        Returns:
            被清理的Key数量
        """
        try:
            with self._lock:
                config = self._config.copy()
                original_count = len(config.get("api_keys", []))
                
                # 过滤出临时Key
                filtered_keys = []
                for key_config in config.get("api_keys", []):
                    key_name = key_config.get("name", "").lower()
                    # 保留不包含 temp, main, key 等临时标识的Key
                    if not any(temp_word in key_name for temp_word in ["temp", "main", "key "]):
                        filtered_keys.append(key_config)
                
                config["api_keys"] = filtered_keys
                self._save_config(config)
                self._config = config
                
                cleaned_count = original_count - len(filtered_keys)
                if cleaned_count > 0:
                    print(f"已清理 {cleaned_count} 个临时API Key")
                
                return cleaned_count
        except Exception as e:
            print(f"清理临时Key失败: {e}")
            return 0
