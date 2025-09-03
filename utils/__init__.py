# Google Nano Utils Package
"""
工具函数模块包，包含图像处理、重试机制、加密等工具函数
"""

from .image_utils import pil_to_base64_data_url, decode_image_from_openrouter_response, tensor_to_pils, pils_to_tensor, validate_and_convert_images, create_size_mismatch_message, get_actual_display_count, save_images_to_output
from .retry_utils import retry_with_backoff, ApiCallError, RateLimitError, QuotaExceededError, AuthenticationError, InvalidRequestError

# 尝试导入加密功能
try:
    from .crypto_utils import get_encryptor, get_config_encryption, simple_encrypt, simple_decrypt, is_encryption_available
    ENCRYPTION_AVAILABLE = True
except ImportError:
    # 如果cryptography库不可用，提供空实现
    def get_encryptor(*args, **kwargs):
        return None
    def get_config_encryption(*args, **kwargs):
        return None
    def simple_encrypt(text):
        return text
    def simple_decrypt(text):
        return text
    def is_encryption_available():
        return False
    ENCRYPTION_AVAILABLE = False

__all__ = [
    'pil_to_base64_data_url', 
    'decode_image_from_openrouter_response', 
    'tensor_to_pils', 
    'pils_to_tensor',
    'validate_and_convert_images',
    'create_size_mismatch_message',
    'get_actual_display_count',
    'save_images_to_output',
    'retry_with_backoff',
    'ApiCallError',
    'RateLimitError',
    'QuotaExceededError', 
    'AuthenticationError',
    'InvalidRequestError',
    'get_encryptor',
    'get_config_manager',
    'simple_encrypt',
    'simple_decrypt',
    'is_encryption_available',
    'ENCRYPTION_AVAILABLE'
]