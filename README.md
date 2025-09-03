# Google Nano - 增强版ComfyUI自定义节点

一个功能增强的ComfyUI自定义节点，专为批量图像生成和API管理而设计。它通过OpenRouter平台调用Google Gemini模型，支持并发处理、多API Key管理、失败重试等企业级功能。
使用时并发可能造成图像显示数量少于生成数量，在COMFYUI输出目录里可以找到所有生成的图片，key_management_mode推荐使用"使用配置文件KEY"模式(因为输入KEY模式最多支持4个，超过时用这个模式)

## ✨ 核心增强功能

### 🚀 并发处理能力
- **并发图片生成控制**: 支持1-10个并发任务同时处理
- **智能负载均衡**: 提升生成效率，实时监控并发状态
- **线程池管理**: 避免API响应超时的负载管理

### 🔑 多API Key管理
- **支持配置多个API Key**: 提高可用性和处理能力
- **三种调度策略**: 
  - 轮换模式（round_robin）：按顺序循环使用
  - 随机模式（random）：随机选择可用Key
  - 加权模式（weighted）：基于配额和成功率智能选择
- **自动故障转移**: 智能切换Key，保障任务稳定性
- **并行模式**: 同时使用多个Key分担负载

### 📊 API Key状态监控
- **实时监控Key的有效性和配额**: 自动检测Key状态
- **自动识别速率限制和配额超限**: 智能处理API限制
- **智能冷却时间管理**: 自动管理Key的恢复时间
- **使用统计显示**: 显示每个Key的使用情况和剩余配额

### 🎯 模型选择功能
- **预设多个Gemini模型选项**: 支持免费和付费模型
- **下拉菜单切换**: 便捷的模型选择界面
- **可扩展的自定义模型列表**: 支持添加新模型

### ⚙️ 配置文件管理
- **config.json集中配置**: 实现配置与代码分离
- **支持配置热重载**: 修改配置无需重启ComfyUI
- **可选的API Key加密存储**: 安全保护敏感信息

### 🛡️ 高级特性
- **失败重试机制**: 可配置的重试次数（0-10次），智能重试策略
- **详细日志系统**: 完整的任务执行日志
- **路径和编码支持**: 完整支持中文路径和多编码格式
- **企业级稳定性**: 完善的错误处理和监控机制

## 🎆 传统功能

### 单图生成模式
- 输入1-4张参考图像
- 提供文本提示词
- 生成符合要求的新图像

### 批量生成模式
- 支持CSV和Excel文件格式
- 文件需包含表头`prompt`的列
- 现在支持并发批量处理，显著提升效率

### 支持的模型
- 默认：`google/gemini-2.5-flash-image-preview:free`
- 新增支持：`google/gemini-2.5-flash-image-preview`
- 可配置其他OpenRouter支持的模型

## 📦 安装要求

### 系统要求
- Python 3.8+
- ComfyUI环境

### 基础依赖
```bash
pip install -r requirements.txt
```

依赖列表：
- `gradio` - Web界面框架
- `pandas` - 数据处理（批量模式）
- `Pillow` - 图像处理
- `openpyxl` - Excel文件支持
- `openai` - OpenAI API客户端
- `requests` - HTTP请求处理
- `cryptography` - 可选，用于API Key加密

### 必需库（ComfyUI环境）
- `torch` - PyTorch深度学习框架
- `numpy` - 数值计算

## 🚀 安装步骤

1. **从GitHub克隆项目**
   ```bash
   # 克隆项目到ComfyUI的custom_nodes目录
   cd /path/to/ComfyUI/custom_nodes/
   git clone https://github.com/annewj023/Comfyui_google_nano.git
   ```
   
   或者手动下载：
   - 访问 https://github.com/annewj023/Comfyui_google_nano.git
   - 下载ZIP文件并解压到ComfyUI的custom_nodes目录

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **获取API密钥**
   - 访问 [OpenRouter](https://openrouter.ai/) 注册账号
   - 获取API密钥

4. **重启ComfyUI**
   - 重启ComfyUI以加载新节点

## 🔧 初始配置

### 自动配置初始化
首次启动时，节点会自动创建 `config.json` 配置文件：

```json
{
  "version": "1.0",
  "api_keys": [],
  "settings": {
    "max_concurrent": 5,
    "scheduling_mode": "round_robin",
    "enable_parallel": false,
    "retry_attempts": 3,
    "log_level": "INFO",
    "encryption_enabled": false
  },
  "models": [
    "google/gemini-2.5-flash-image-preview:free",
    "google/gemini-2.5-flash-image-preview",
    "google/gemini-1.5-pro:free",
    "google/gemini-1.5-pro"
  ]
}
```

### API Key配置方式

#### 方式1：直接输入（向后兼容）
- 在节点中直接输入API Key
- 适用于临时使用或单Key场景

#### 方式2：配置文件管理（推荐）
通过修改 `config.json` 添加多个API Key：

```json
{
  "api_keys": [
    {
      "id": "key1",
      "name": "key1",
      "value": "添加你的KEY",
      "encrypted": true,
      "status": "available",
      "stats": {
        "daily_remaining": 50,
        "last_used": null,
        "success_count": 0,
        "error_count": 0,
        "total_requests": 0,
        "usage": 0.0,
        "limit": null,
        "remaining": "unlimited",
        "is_free_tier": true,
        "label": "",
        "last_checked": ""
      },
      "cooldown_until": null,
      "created_at": ""
    }
  ]
}
```

## 📖 使用方法

### 在ComfyUI中使用

1. **添加节点**
   - 在ComfyUI工作流中搜索"google nano"
   - 添加"GoogleNanoNode"节点

2. **配置新版参数**
   - `api_key`: 输入OpenRouter API密钥
   - `prompt`: 输入生成提示词
   - `file_path`: 批量文件路径（批量模式）
   - `model`: 选择模型（默认为Gemini 2.5）
   - `max_concurrent`: 最大并发数（1-10）
   - `scheduling_mode`: Key调度模式（轮换/随机/加权）
   - `enable_parallel`: 启用并行模式
   - `max_retries`: 最大重试次数（0-10）
   - `enable_detailed_logs`: 启用详细日志
   - `site_url/site_name`: 可选的站点信息

3. **连接图像输入**
   - 将参考图像连接到`image1`-`image4`端口
   - 至少需要一张参考图像

4. **运行生成**
   - 执行工作流开始生成图像
   - 查看详细日志输出和性能统计

### 新版节点参数说明

#### 必填参数
- `api_key` (STRING): OpenRouter API密钥（可留空使用配置文件中的Key）

#### 可选参数
- `prompt` (STRING): 文本提示词，用于单图生成
- `file_path` (STRING): 批量文件路径（CSV/Excel），支持中文路径和带空格的路径
- `model` (DROPDOWN): 使用的AI模型名称
- `max_concurrent` (INT): 最大并发任务数量（1-10）
- `scheduling_mode` (DROPDOWN): API Key调度模式（轮换/随机/加权）
- `enable_parallel` (BOOLEAN): 是否启用并行模式（同时使用多个API Key）
- `max_retries` (INT): API调用失败时的最大重试次数
- `enable_detailed_logs` (BOOLEAN): 启用详细的任务执行日志
- `site_url` (STRING): 站点URL（用于API统计）
- `site_name` (STRING): 站点名称（用于API统计）
- `image1-4` (IMAGE): 参考图像输入

### 批量模式文件格式

#### 文件路径格式说明
支持以下路径格式：
- **绝对路径**: `C:\Users\用户名\Documents\prompts.csv`
- **相对路径**: `./data/prompts.xlsx`
- **中文路径**: `D:\项目文件\提示词列表.csv`
- **带空格路径**: `"C:\Program Files\Data\prompts.csv"`
- **带双引号路径**: 系统会自动处理路径两端的双引号

#### 路径示例
```
# Windows系统路径示例
C:\Users\张三\Desktop\图像生成\prompts.csv
D:\AI项目\批量提示词.xlsx
"C:\Program Files\ComfyUI\data\prompts.csv"

# 相对路径示例
./prompts.csv
../data/batch_prompts.xlsx
```

#### 文件内容格式

**重要提醒：**
- CSV文件建议使用UTF-8编码保存，以确保中文字符正确显示
- 如果遇到中文乱码，可尝试用GBK编码保存CSV文件
- Excel文件会自动处理编码问题

**CSV格式示例:**
```csv
prompt
"生成一张蓝色的天空"
"画一只可爱的小猫"
"创建现代建筑设计"
"绘制中国古典园林景观"
"设计未来科技城市"
```

**Excel格式示例:**
| prompt |
|--------|
| 生成一张蓝色的天空 |
| 画一只可爱的小猫 |
| 创建现代建筑设计 |
| 绘制中国古典园林景观 |
| 设计未来科技城市 |

## 📊 监控和日志

### 日志功能
- 自动创建 `logs/` 目录
- 任务执行详情记录
- API调用统计信息
- 错误和重试记录
- 实时性能指标显示

### 日志导出
系统支持导出以下格式的日志：
- 任务概览CSV
- API调用详情CSV
- JSON格式完整日志

### 状态监控
- API Key可用性实时显示
- 并发任务数量监控
- 成功率和错误统计
- 配额使用情况显示

## 🔐 安全特性

### API Key保护
- 可选的加密存储功能
- 基于机器特征的密钥派生
- 避免明文泄露风险

### 错误处理
- 完善的异常捕获机制
- 详细的错误诊断信息
- 优雅的降级处理

## 📈 性能优化

### 并发处理
- 线程池管理并发任务
- 智能任务分发算法
- 内存和CPU使用优化

### API管理
- 连接池复用
- 智能Key选择策略
- 自动负载均衡

### 性能指标
- 平均任务持续时间
- 每任务图片生成数
- API调用成功率
- 系统资源使用情况

## ⚠️ 注意事项

### OpenRouter API限制规则
- **免费模型限制**：
  - 每分钟最多20次请求
  - 购买低于10学分：每日50次免费请求
  - 购买至少10学分：每日1000次免费请求
- **DDoS保护**：Cloudflare会阻止过度使用的请求

### 使用建议
- 合理设置并发数量，避免触发限制
- 监控API配额使用情况
- 配置多个API Key以提高可用性
- 定期查看日志文件进行问题排查

## 🛠️ 故障排除

### 常见问题

**1. "未安装 openai 库"**
- 解决：`pip install openai`

**2. "没有可用的API Key"**
- 检查配置文件或直接输入Key
- 确认Key的有效性和配额

**3. "文件路径不存在"**
- 支持中文路径和带空格路径
- 检查文件格式（CSV/Excel）
- 确认文件包含`prompt`列

**4. 加密功能不可用**
- 安装：`pip install cryptography`
- 或禁用加密功能使用明文存储

**5. 并发数过高导致限制**
- 降低`max_concurrent`参数
- 检查API Key的配额限制
- 使用多个API Key分散负载

### 调试技巧
1. 启用详细日志查看执行过程
2. 检查 `logs/` 目录中的日志文件
3. 使用单个API Key测试基础功能
4. 逐步增加并发数量进行测试
5. 监控系统资源使用情况

## 📚 技术架构

### 模块化设计
- `managers/`: 配置、API Key、日志管理
- `utils/`: 图像处理、重试机制、加密工具
- `google_nano.py`: 主节点实现

### 设计原则
- 向后兼容性保证
- 模块化和可扩展性
- 错误处理和日志记录
- 性能和安全平衡

## ⚙️ 节点参数说明

### 必需参数
- `api_key` (STRING): OpenRouter API密钥

### 可选参数
- `prompt` (STRING): 文本提示词，用于单图生成
- `file_path` (STRING): 批量文件路径（CSV/Excel），支持中文路径和带空格的路径，可使用绝对或相对路径
- `site_url` (STRING): 站点URL（用于API统计）
- `site_name` (STRING): 站点名称（用于API统计）
- `model` (STRING): 使用的AI模型名称
- `image1-4` (IMAGE): 参考图像输入（至少需要一张）


## 📄 许可证

本项目遵循开源许可证，具体许可证信息请查看项目根目录。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📞 支持

如有问题，请在项目Issue页面提交问题报告。

---

**注意**: 使用本工具需要有效的OpenRouter API密钥，API调用可能产生费用，请合理使用。

