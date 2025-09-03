"""
加密工具模块 - 提供API Key的加密和解密功能
"""

import base64
import hashlib
import os
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SimpleEncryptor:
    """简单的加密器，用于API Key的安全存储"""
    
    def __init__(self, password: str = None):
        """
        初始化加密器
        
        Args:
            password: 主密码，如果为None则使用机器特征生成
        """
        if password is None:
            password = self._generate_machine_key()
        
        self.password = password.encode() if isinstance(password, str) else password
        self._fernet = None
    
    def _generate_machine_key(self) -> str:
        """基于机器特征生成密钥"""
        try:
            import platform
            import getpass
            
            # 收集机器特征
            features = [
                platform.machine(),
                platform.processor(),
                platform.system(),
                platform.node(),
                getpass.getuser()
            ]
            
            # 组合特征并生成哈希
            combined = "".join(str(f) for f in features if f)
            if not combined:
                combined = "google_nano_default_key"
            
            return hashlib.sha256(combined.encode()).hexdigest()[:32]
        except Exception:
            # 如果无法获取机器特征，使用默认值
            return "google_nano_fallback_key_2024"
    
    def _get_fernet(self) -> Fernet:
        """获取Fernet加密实例"""
        if self._fernet is None:
            # 使用PBKDF2从密码派生密钥
            salt = b'google_nano_salt_2024'  # 固定盐值，确保一致性
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.password))
            self._fernet = Fernet(key)
        
        return self._fernet
    
    def encrypt(self, plaintext: str) -> str:
        """
        加密字符串
        
        Args:
            plaintext: 要加密的明文
            
        Returns:
            加密后的base64字符串
        """
        if not plaintext:
            return ""
        
        try:
            fernet = self._get_fernet()
            encrypted_bytes = fernet.encrypt(plaintext.encode())
            return base64.b64encode(encrypted_bytes).decode()
        except Exception as e:
            print(f"加密失败: {e}")
            return plaintext  # 如果加密失败，返回原文
    
    def decrypt(self, ciphertext: str) -> str:
        """
        解密字符串
        
        Args:
            ciphertext: 要解密的密文（base64编码）
            
        Returns:
            解密后的明文
        """
        if not ciphertext:
            return ""
        
        try:
            fernet = self._get_fernet()
            encrypted_bytes = base64.b64decode(ciphertext.encode())
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            print(f"解密失败: {e}")
            return ciphertext  # 如果解密失败，返回原文（可能是明文存储）
    
    def is_encrypted(self, text: str) -> bool:
        """
        检查文本是否已加密
        
        Args:
            text: 要检查的文本
            
        Returns:
            是否为加密文本
        """
        if not text:
            return False
        
        try:
            # 尝试解密，如果成功且结果不同则说明是加密的
            decrypted = self.decrypt(text)
            return decrypted != text and len(decrypted) > 0
        except Exception:
            return False


class ConfigEncryption:
    """配置文件加密管理器"""
    
    def __init__(self, encryptor: SimpleEncryptor):
        """
        初始化配置加密管理器
        
        Args:
            encryptor: 加密器实例
        """
        self.encryptor = encryptor
    
    def encrypt_api_keys(self, config: dict) -> dict:
        """
        加密配置中的API Keys
        
        Args:
            config: 配置字典
            
        Returns:
            加密后的配置字典
        """
        if not config or "api_keys" not in config:
            return config
        
        # 创建配置副本
        encrypted_config = config.copy()
        encrypted_config["api_keys"] = []
        
        for key_config in config.get("api_keys", []):
            encrypted_key = key_config.copy()
            
            # 加密API Key值
            api_key_value = key_config.get("value", "")
            if api_key_value and not self.encryptor.is_encrypted(api_key_value):
                encrypted_key["value"] = self.encryptor.encrypt(api_key_value)
                encrypted_key["encrypted"] = True
            
            encrypted_config["api_keys"].append(encrypted_key)
        
        # 标记加密状态
        encrypted_config.setdefault("settings", {})["encryption_enabled"] = True
        
        return encrypted_config
    
    def decrypt_api_keys(self, config: dict) -> dict:
        """
        解密配置中的API Keys
        
        Args:
            config: 加密的配置字典
            
        Returns:
            解密后的配置字典
        """
        if not config or "api_keys" not in config:
            return config
        
        # 创建配置副本
        decrypted_config = config.copy()
        decrypted_config["api_keys"] = []
        
        for key_config in config.get("api_keys", []):
            decrypted_key = key_config.copy()
            
            # 解密API Key值
            if key_config.get("encrypted", False):
                encrypted_value = key_config.get("value", "")
                if encrypted_value:
                    decrypted_key["value"] = self.encryptor.decrypt(encrypted_value)
                    decrypted_key["encrypted"] = False
            
            decrypted_config["api_keys"].append(decrypted_key)
        
        return decrypted_config
    
    def get_decrypted_key_value(self, key_config: dict) -> str:
        """
        获取解密后的API Key值
        
        Args:
            key_config: API Key配置
            
        Returns:
            解密后的API Key值
        """
        if not key_config:
            return ""
        
        api_key_value = key_config.get("value", "")
        if key_config.get("encrypted", False) and api_key_value:
            return self.encryptor.decrypt(api_key_value)
        
        return api_key_value


# 全局加密器实例（单例模式）
_global_encryptor: Optional[SimpleEncryptor] = None
_global_config_encryption: Optional[ConfigEncryption] = None


def get_encryptor(password: str = None) -> SimpleEncryptor:
    """
    获取全局加密器实例
    
    Args:
        password: 主密码
        
    Returns:
        加密器实例
    """
    global _global_encryptor
    
    if _global_encryptor is None:
        _global_encryptor = SimpleEncryptor(password)
    
    return _global_encryptor


def get_config_encryption(password: str = None) -> ConfigEncryption:
    """
    获取全局配置加密管理器实例
    
    Args:
        password: 主密码
        
    Returns:
        配置加密管理器实例
    """
    global _global_config_encryption
    
    if _global_config_encryption is None:
        encryptor = get_encryptor(password)
        _global_config_encryption = ConfigEncryption(encryptor)
    
    return _global_config_encryption


def simple_encrypt(text: str) -> str:
    """
    简单加密函数（使用默认加密器）
    
    Args:
        text: 要加密的文本
        
    Returns:
        加密后的文本
    """
    encryptor = get_encryptor()
    return encryptor.encrypt(text)


def simple_decrypt(text: str) -> str:
    """
    简单解密函数（使用默认加密器）
    
    Args:
        text: 要解密的文本
        
    Returns:
        解密后的文本
    """
    encryptor = get_encryptor()
    return encryptor.decrypt(text)


# 检查是否可用加密功能
def is_encryption_available() -> bool:
    """检查是否可以使用加密功能"""
    try:
        from cryptography.fernet import Fernet
        return True
    except ImportError:
        return False