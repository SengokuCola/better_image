# Better Image

Better Image 是一个 MaiBot 插件，提供互联网搜图、历史消息图片提取和裁切放大能力。

当前版本：`1.1.5`，支持 MaiBot `1.3.0–1.3.99` 和插件 SDK `2.9.x`。

## 功能

- `image_search`：从互联网搜索图片并以工具图片结果返回；支持 DuckDuckGo、Bing 和 Safebooru。
- `image_crop`：从历史消息中读取图片，按比例或像素裁切、放大，并保存到插件上下文。
- 支持在 `config.toml` 中分别启用或关闭上述两个工具。

## 配置

仓库与发布包只提供 `config.template.toml` 配置模板。首次加载时插件会生成 `config.toml`，也可以复制模板为 `config.toml` 后修改；实际配置文件不纳入 Git。

```toml
[plugin]
enabled = true
config_version = "1.4.0"

[tools]
search = true
crop = true
```

## 使用说明

安装后启用插件即可使用工具。`image_search` 的 `source` 参数可设为 `auto`、`duckduckgo`、`bing` 或 `safebooru`。Safebooru 使用英文标签搜索，例如 `hatsune_miku`；`auto` 会在其他图源结果不足时查询 Safebooru。互联网搜图依赖公开接口和页面，结果可用性会受网络环境和搜索源限制。

## 更新记录

### 1.1.5

- 发布包只包含 `config.template.toml`，停止跟踪实际 `config.toml`。
- 主程序兼容范围修正为 MaiBot `1.3.0–1.3.99`。

### 1.1.4

- 适配 MaiBot 1.3.4 / 插件 SDK 2.9.0 的工具简要描述与详细说明。
- 图片下载解析、裁切和放大在线程中执行，避免阻塞插件事件循环。
- 消息查询失败时保留具体错误，补齐 Pillow 依赖声明。
- 本次发布同时包含此前本地更新：新增 Safebooru 图源，工具改为 `image_search` / `image_crop`，移除图片变换工具。

### 1.1.3

- 适配 MaiBot 1.2.x。

### 1.1.2

- 适配 MaiBot 1.1.x。

### 1.1.1

#### 用户感知功能

- 新增 `better_image_transform`，支持图片翻转、旋转、等比缩放和非等比缩放。
- 将裁切工具更名为 `better_image_crop`，强化工具用途表达。
- 移除上下文图片发送工具，插件专注于图片搜索、裁切和变换。

#### 开发侧

- 优化搜图关键词降级与 Bing 图片结果解析。
- 将配置项调整为 `search`、`crop`、`transform`，兼容旧版 `get` 配置迁移。

## 许可证

MIT
