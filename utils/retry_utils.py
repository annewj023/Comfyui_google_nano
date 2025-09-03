"""
重试机制工具函数 - 提供API调用的重试功能和错误处理
"""

import time
import random
import functools
from typing import Callable, Any, Optional, Dict
import logging


class ApiCallError(Exception):
    """API调用错误基类"""
    def __init__(self, message: str, error_type: str = "unknown", response_data: Dict = None):
        super().__init__(message)
        self.error_type = error_type
        self.response_data = response_data or {}


class RateLimitError(ApiCallError):
    """速率限制错误"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message, "rate_limit")
        self.retry_after = retry_after


class QuotaExceededError(ApiCallError):
    """配额超限错误"""
    def __init__(self, message: str):
        super().__init__(message, "quota_exceeded")


class AuthenticationError(ApiCallError):
    """认证错误"""
    def __init__(self, message: str):
        super().__init__(message, "authentication")


class InvalidRequestError(ApiCallError):
    """无效请求错误"""
    def __init__(self, message: str):
        super().__init__(message, "invalid_request")


def parse_api_error(error: Exception, response_data: Dict = None) -> ApiCallError:
    """
    解析API错误，转换为具体的错误类型
    
    Args:
        error: 原始错误对象
        response_data: API响应数据
        
    Returns:
        具体的ApiCallError子类实例
    """
    error_message = str(error).lower()
    
    # 检查是否为速率限制错误
    if any(keyword in error_message for keyword in ["rate limit", "429", "too many requests"]):
        # 尝试解析retry-after头
        retry_after = None
        if response_data and "headers" in response_data:
            retry_after_header = response_data["headers"].get("retry-after")
            if retry_after_header:
                try:
                    retry_after = int(retry_after_header)
                except (ValueError, TypeError):
                    pass
        return RateLimitError(str(error), retry_after)
    
    # 检查是否为配额超限错误
    if any(keyword in error_message for keyword in ["quota", "limit exceeded", "usage limit"]):
        return QuotaExceededError(str(error))
    
    # 检查是否为认证错误
    if any(keyword in error_message for keyword in ["auth", "401", "403", "unauthorized", "forbidden", "api key"]):
        return AuthenticationError(str(error))
    
    # 检查是否为无效请求错误
    if any(keyword in error_message for keyword in ["400", "bad request", "invalid", "malformed"]):
        return InvalidRequestError(str(error))
    
    # 默认为一般API错误
    return ApiCallError(str(error), "general", response_data)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on_exceptions: tuple = (Exception,),
    stop_on_exceptions: tuple = (AuthenticationError, InvalidRequestError),
    logger: Optional[logging.Logger] = None
):
    """
    重试装饰器，支持指数退避和抖动
    
    Args:
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数退避的基数
        jitter: 是否添加随机抖动
        retry_on_exceptions: 需要重试的异常类型
        stop_on_exceptions: 不重试的异常类型
        logger: 日志记录器
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # 解析API错误
                    if hasattr(e, 'response'):
                        api_error = parse_api_error(e, getattr(e, 'response', {}))
                    else:
                        api_error = parse_api_error(e)
                    
                    # 检查是否应该停止重试
                    if isinstance(api_error, stop_on_exceptions):
                        if logger:
                            logger.error(f"遇到不可重试的错误，停止重试: {api_error}")
                        raise api_error
                    
                    # 检查是否应该重试
                    if not isinstance(e, retry_on_exceptions):
                        if logger:
                            logger.error(f"遇到不在重试列表中的错误: {type(e).__name__}")
                        raise e
                    
                    # 如果这是最后一次尝试，直接抛出异常
                    if attempt == max_attempts - 1:
                        if logger:
                            logger.error(f"达到最大重试次数 ({max_attempts})，最后的错误: {api_error}")
                        raise api_error
                    
                    # 计算延迟时间
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # 添加抖动
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # 0.5-1.0倍的随机系数
                    
                    # 对于速率限制错误，使用API建议的延迟时间
                    if isinstance(api_error, RateLimitError) and api_error.retry_after:
                        delay = max(delay, api_error.retry_after)
                    
                    if logger:
                        logger.warning(f"第 {attempt + 1} 次尝试失败: {api_error}, {delay:.2f}秒后重试")
                    
                    time.sleep(delay)
            
            # 理论上不会到达这里，但为了安全起见
            raise last_exception
        
        return wrapper
    return decorator


def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    创建断路器，防止持续失败的API调用
    
    Args:
        failure_threshold: 失败阈值
        recovery_timeout: 恢复超时时间（秒）
        expected_exception: 触发断路器的异常类型
        
    Returns:
        断路器装饰器
    """
    def decorator(func: Callable) -> Callable:
        func._failure_count = 0
        func._last_failure_time = 0
        func._circuit_open = False
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # 检查断路器是否应该重置
            if func._circuit_open and (now - func._last_failure_time) > recovery_timeout:
                func._circuit_open = False
                func._failure_count = 0
            
            # 如果断路器开启，直接抛出异常
            if func._circuit_open:
                raise ApiCallError(f"断路器开启，请 {recovery_timeout} 秒后重试", "circuit_breaker")
            
            try:
                result = func(*args, **kwargs)
                # 成功调用，重置失败计数
                func._failure_count = 0
                return result
            except expected_exception as e:
                func._failure_count += 1
                func._last_failure_time = now
                
                # 检查是否达到失败阈值
                if func._failure_count >= failure_threshold:
                    func._circuit_open = True
                
                raise e
        
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    超时装饰器
    
    Args:
        timeout_seconds: 超时时间（秒）
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"函数执行超时 ({timeout_seconds} 秒)")
            
            # 设置超时信号
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 恢复原始信号处理器
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


def safe_api_call(func: Callable, *args, **kwargs) -> tuple[Any, Optional[Exception]]:
    """
    安全的API调用包装器，返回结果和错误
    
    Args:
        func: 要调用的函数
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        tuple[结果, 错误]: 如果成功返回(结果, None)，如果失败返回(None, 错误)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e


def batch_retry_wrapper(
    items: list,
    process_func: Callable,
    max_workers: int = 3,
    retry_failed: bool = True,
    **retry_kwargs
) -> tuple[list, list]:
    """
    批量处理带重试的包装器
    
    Args:
        items: 要处理的项目列表
        process_func: 处理函数
        max_workers: 最大并发数
        retry_failed: 是否重试失败的项目
        **retry_kwargs: 重试参数
        
    Returns:
        tuple[成功结果列表, 失败项目列表]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 应用重试装饰器
    if retry_failed:
        process_func = retry_with_backoff(**retry_kwargs)(process_func)
    
    successful_results = []
    failed_items = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_func, item): item
            for item in items
        }
        
        # 收集结果
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                successful_results.append((item, result))
            except Exception as e:
                failed_items.append((item, e))
    
    return successful_results, failed_items