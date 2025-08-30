# Google Nano - AI图像生成节点

一个基于OpenRouter的ComfyUI自定义节点，支持通过OpenAI API调用Google Gemini模型，根据参考图像和提示词生成新的图像。

## 🌟 主要功能

- **AI图像生成**: 使用Google Gemini 2.5 Flash Image Preview模型生成图像
- **多图参考**: 支持最多4张参考图像同时输入
- **批量处理**: 支持通过CSV/Excel文件批量生成图像
- **ComfyUI集成**: 完全兼容ComfyUI工作流
- **灵活配置**: 支持自定义模型、API密钥等参数

## 📋 功能特性

### 单图生成模式
- 输入1-4张参考图像
- 提供文本提示词
- 生成符合要求的新图像

### 批量生成模式
- 支持CSV和Excel文件格式
- 文件需包含表头`prompt`的列
- 自动遍历所有提示词进行批量生成

### 支持的模型
- 默认：`google/gemini-2.5-flash-image-preview:free`
- 可配置其他OpenRouter支持的模型

## 🛠️ 安装要求

### 系统要求
- Python 3.8+
- ComfyUI环境

### 依赖包
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

## 📖 使用方法

### 在ComfyUI中使用

1. **添加节点**
   - 在ComfyUI工作流中搜索"google nano"
   - 添加"GoogleNanoNode"节点

2. **配置参数**
   - `api_key`: 输入OpenRouter API密钥
   - `prompt`: 输入生成提示词（单图模式）
   - `file_path`: 批量文件路径（批量模式）
   - `model`: 选择模型（默认为Gemini 2.5）
   - `site_url/site_name`: 可选的站点信息

3. **连接图像输入**
   - 将参考图像连接到`image1`-`image4`端口
   - 至少需要一张参考图像

4. **运行生成**
   - 执行工作流开始生成图像

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