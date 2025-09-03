"""
任务日志记录器 - 负责详细记录任务执行日志，支持导出和分析
"""

import logging
import os
import json
import csv
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskLogger:
    """任务日志记录器"""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = None):
        """
        初始化任务日志记录器
        
        Args:
            log_level: 日志级别
            log_dir: 日志文件目录，如果为None则使用默认目录
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        if log_dir is None:
            # 默认日志目录为当前模块所在目录的logs子目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.log_dir = os.path.join(os.path.dirname(current_dir), "logs")
        else:
            self.log_dir = log_dir
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._lock = threading.RLock()
        self._task_logs = []  # 任务日志列表
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_pending = False  # 是否有待保存的更改
        self._last_save_time = 0  # 上次保存时间
        self._save_interval = 2.0  # 保存间隔（秒），减少频繁I/O
        
        # 先设置日志文件路径
        self.task_log_file = os.path.join(self.log_dir, f"tasks_{self._session_id}.json")
        self.general_log_file = os.path.join(self.log_dir, f"general_{self._session_id}.log")
        
        # 然后设置日志格式
        self._setup_logging()
        
        # 初始化时清空任务日志文件（覆盖模式）
        self._save_task_logs()
        
        self.logger.info(f"TaskLogger初始化完成，会话ID: {self._session_id}")
    
    def _setup_logging(self):
        """设置日志配置"""
        # 创建logger
        self.logger = logging.getLogger(f"GoogleNano_{self._session_id}")
        self.logger.setLevel(self.log_level)
        
        # 清除现有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 文件处理器（使用覆盖模式 'w'）
        file_handler = logging.FileHandler(
            self.general_log_file,
            mode='w',  # 覆盖模式，而不是默认的追加模式 'a'
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def create_task_log(self, task_id: str, task_type: str, task_data: Dict[str, Any]) -> str:
        """
        创建新的任务日志记录
        
        Args:
            task_id: 任务ID
            task_type: 任务类型（single, batch等）
            task_data: 任务数据
            
        Returns:
            任务日志ID
        """
        with self._lock:
            task_log = {
                "task_id": task_id,
                "log_id": f"{self._session_id}_{len(self._task_logs)}",
                "task_type": task_type,
                "status": TaskStatus.PENDING.value,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "task_data": task_data,
                "api_calls": [],
                "errors": [],
                "metrics": {
                    "start_time": None,
                    "end_time": None,
                    "duration": None,
                    "total_api_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "retry_count": 0,
                    "images_generated": 0
                },
                "key_usage": {}  # 记录各API Key的使用情况
            }
            
            self._task_logs.append(task_log)
            self._save_task_logs()
            
            self.logger.info(f"创建任务日志: {task_log['log_id']} (类型: {task_type})")
            return task_log["log_id"]
    
    def update_task_status(self, log_id: str, status: TaskStatus, message: str = ""):
        """更新任务状态"""
        with self._lock:
            task_log = self._find_task_log(log_id)
            if task_log:
                old_status = task_log["status"]
                task_log["status"] = status.value
                task_log["updated_at"] = datetime.now().isoformat()
                
                # 记录时间戳
                current_time = datetime.now().isoformat()
                if status == TaskStatus.RUNNING and not task_log["metrics"]["start_time"]:
                    task_log["metrics"]["start_time"] = current_time
                elif status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task_log["metrics"]["end_time"] = current_time
                    # 计算持续时间
                    if task_log["metrics"]["start_time"]:
                        start_time = datetime.fromisoformat(task_log["metrics"]["start_time"])
                        end_time = datetime.fromisoformat(current_time)
                        duration = (end_time - start_time).total_seconds()
                        task_log["metrics"]["duration"] = duration
                
                self._save_task_logs()
                
                log_message = f"任务 {log_id} 状态更新: {old_status} -> {status.value}"
                if message:
                    log_message += f" - {message}"
                
                if status == TaskStatus.FAILED:
                    self.logger.error(log_message)
                elif status == TaskStatus.SUCCESS:
                    self.logger.info(log_message)
                else:
                    self.logger.info(log_message)
    
    def log_api_call(self, log_id: str, api_key_id: str, model: str, 
                     request_data: Dict[str, Any], response_data: Dict[str, Any] = None, 
                     error: str = None, duration: float = None):
        """
        记录API调用详情
        
        Args:
            log_id: 任务日志ID
            api_key_id: 使用的API Key ID
            model: 使用的模型
            request_data: 请求数据
            response_data: 响应数据
            error: 错误信息
            duration: 调用耗时（秒）
        """
        with self._lock:
            task_log = self._find_task_log(log_id)
            if task_log:
                api_call_log = {
                    "call_id": f"{log_id}_call_{len(task_log['api_calls'])}",
                    "timestamp": datetime.now().isoformat(),
                    "api_key_id": api_key_id,
                    "model": model,
                    "duration": duration,
                    "success": error is None,
                    "request_data": {
                        "prompt": request_data.get("prompt", ""),
                        "image_count": len(request_data.get("images", [])),
                        "model": model
                    },
                    "response_data": response_data,
                    "error": error
                }
                
                task_log["api_calls"].append(api_call_log)
                
                # 更新metrics
                metrics = task_log["metrics"]
                metrics["total_api_calls"] += 1
                
                if error:
                    metrics["failed_calls"] += 1
                    self.log_error(log_id, f"API调用失败: {error}")
                else:
                    metrics["successful_calls"] += 1
                    # 统计生成的图片数量
                    if response_data and "images" in response_data:
                        metrics["images_generated"] += len(response_data["images"])
                
                # 更新Key使用统计
                key_usage = task_log["key_usage"]
                if api_key_id not in key_usage:
                    key_usage[api_key_id] = {"calls": 0, "successes": 0, "errors": 0}
                
                key_usage[api_key_id]["calls"] += 1
                if error:
                    key_usage[api_key_id]["errors"] += 1
                else:
                    key_usage[api_key_id]["successes"] += 1
                
                self._save_task_logs()
                
                if error:
                    self.logger.error(f"API调用失败 [{api_key_id}]: {error}")
                else:
                    self.logger.info(f"API调用成功 [{api_key_id}] - 模型: {model}, 耗时: {duration:.2f}s")
    
    def log_retry(self, log_id: str, attempt: int, reason: str):
        """记录重试信息"""
        with self._lock:
            task_log = self._find_task_log(log_id)
            if task_log:
                task_log["metrics"]["retry_count"] = attempt
                self.logger.warning(f"任务 {log_id} 第 {attempt} 次重试: {reason}")
                self._save_task_logs()
    
    def log_info(self, log_id: str, message: str):
        """
        记录信息日志
        
        Args:
            log_id: 任务日志ID
            message: 信息内容
        """
        self.logger.info(f"[{log_id}] {message}")
    
    def log_error(self, log_id: str, error_message: str, error_type: str = "general"):
        """记录错误信息"""
        with self._lock:
            task_log = self._find_task_log(log_id)
            if task_log:
                error_log = {
                    "timestamp": datetime.now().isoformat(),
                    "type": error_type,
                    "message": error_message
                }
                task_log["errors"].append(error_log)
                self._save_task_logs()
                
                self.logger.error(f"任务 {log_id} 错误 [{error_type}]: {error_message}")
    
    def _find_task_log(self, log_id: str) -> Optional[Dict[str, Any]]:
        """查找任务日志"""
        for task_log in self._task_logs:
            if task_log["log_id"] == log_id:
                return task_log
        return None
    
    def _save_task_logs(self, force=False):
        """保存任务日志到文件（优化了I/O频率）"""
        current_time = time.time()
        
        # 如果不是强制保存且距离上次保存时间太短，则跳过
        if not force and (current_time - self._last_save_time) < self._save_interval:
            self._save_pending = True
            return
            
        try:
            with open(self.task_log_file, 'w', encoding='utf-8') as f:
                json.dump(self._task_logs, f, indent=2, ensure_ascii=False)
            self._last_save_time = current_time
            self._save_pending = False
        except Exception as e:
            self.logger.error(f"保存任务日志失败: {e}")
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        with self._lock:
            total_tasks = len(self._task_logs)
            if total_tasks == 0:
                return {"total_tasks": 0, "message": "暂无任务数据"}
            
            status_count = {}
            total_api_calls = 0
            total_successes = 0
            total_failures = 0
            total_images = 0
            total_duration = 0
            
            for task_log in self._task_logs:
                status = task_log["status"]
                status_count[status] = status_count.get(status, 0) + 1
                
                metrics = task_log["metrics"]
                total_api_calls += metrics.get("total_api_calls", 0)
                total_successes += metrics.get("successful_calls", 0)
                total_failures += metrics.get("failed_calls", 0)
                total_images += metrics.get("images_generated", 0)
                
                if metrics.get("duration"):
                    total_duration += metrics["duration"]
            
            success_rate = (total_successes / total_api_calls * 100) if total_api_calls > 0 else 0
            avg_duration = total_duration / total_tasks if total_tasks > 0 else 0
            
            return {
                "session_id": self._session_id,
                "total_tasks": total_tasks,
                "status_breakdown": status_count,
                "api_statistics": {
                    "total_api_calls": total_api_calls,
                    "successful_calls": total_successes,
                    "failed_calls": total_failures,
                    "success_rate": round(success_rate, 2)
                },
                "performance": {
                    "total_images_generated": total_images,
                    "average_task_duration": round(avg_duration, 2),
                    "images_per_task": round(total_images / total_tasks, 2) if total_tasks > 0 else 0
                }
            }
    
    def export_logs_csv(self, output_path: str = None) -> str:
        """导出日志为CSV格式"""
        if output_path is None:
            output_path = os.path.join(self.log_dir, f"task_export_{self._session_id}.csv")
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = [
                    'task_id', 'log_id', 'task_type', 'status', 'created_at', 'updated_at',
                    'duration', 'total_api_calls', 'successful_calls', 'failed_calls',
                    'retry_count', 'images_generated', 'error_count'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                with self._lock:
                    for task_log in self._task_logs:
                        metrics = task_log["metrics"]
                        row = {
                            'task_id': task_log["task_id"],
                            'log_id': task_log["log_id"],
                            'task_type': task_log["task_type"],
                            'status': task_log["status"],
                            'created_at': task_log["created_at"],
                            'updated_at': task_log["updated_at"],
                            'duration': metrics.get("duration", ""),
                            'total_api_calls': metrics.get("total_api_calls", 0),
                            'successful_calls': metrics.get("successful_calls", 0),
                            'failed_calls': metrics.get("failed_calls", 0),
                            'retry_count': metrics.get("retry_count", 0),
                            'images_generated': metrics.get("images_generated", 0),
                            'error_count': len(task_log["errors"])
                        }
                        writer.writerow(row)
            
            self.logger.info(f"日志已导出至CSV: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"导出CSV失败: {e}")
            return ""
    
    def export_api_calls_csv(self, output_path: str = None) -> str:
        """导出API调用详情为CSV格式"""
        if output_path is None:
            output_path = os.path.join(self.log_dir, f"api_calls_{self._session_id}.csv")
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = [
                    'task_id', 'call_id', 'timestamp', 'api_key_id', 'model',
                    'duration', 'success', 'prompt_length', 'image_count', 'error'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                with self._lock:
                    for task_log in self._task_logs:
                        for api_call in task_log["api_calls"]:
                            request_data = api_call.get("request_data", {})
                            row = {
                                'task_id': task_log["task_id"],
                                'call_id': api_call["call_id"],
                                'timestamp': api_call["timestamp"],
                                'api_key_id': api_call["api_key_id"],
                                'model': api_call["model"],
                                'duration': api_call.get("duration", ""),
                                'success': api_call["success"],
                                'prompt_length': len(request_data.get("prompt", "")),
                                'image_count': request_data.get("image_count", 0),
                                'error': api_call.get("error", "")
                            }
                            writer.writerow(row)
            
            self.logger.info(f"API调用详情已导出至CSV: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"导出API调用CSV失败: {e}")
            return ""
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """清理旧日志文件"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    if file_mtime < cutoff_time:
                        os.remove(file_path)
                        self.logger.info(f"删除旧日志文件: {filename}")
        except Exception as e:
            self.logger.error(f"清理旧日志失败: {e}")
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的错误信息"""
        recent_errors = []
        
        with self._lock:
            for task_log in reversed(self._task_logs):  # 从最新的开始
                for error in reversed(task_log["errors"]):
                    if len(recent_errors) >= limit:
                        break
                    
                    recent_errors.append({
                        "task_id": task_log["task_id"],
                        "timestamp": error["timestamp"],
                        "type": error["type"],
                        "message": error["message"]
                    })
                
                if len(recent_errors) >= limit:
                    break
        
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            # 强制保存所有待存日志
            if hasattr(self, '_save_pending') and self._save_pending:
                self._save_task_logs(force=True)
        except:
            pass
    
    def force_cleanup(self):
        """强制清理所有资源"""
        try:
            with self._lock:
                if self._save_pending:
                    self._save_task_logs(force=True)
                # 关闭所有文件句柄
                for handler in self.logger.handlers[:]:
                    try:
                        handler.close()
                        self.logger.removeHandler(handler)
                    except:
                        pass
        except:
            pass