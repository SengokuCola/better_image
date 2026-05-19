# Better Image

Better Image 是一个 MaiBot 插件，提供互联网搜图、历史消息图片提取、裁切放大和图片变换能力。

当前版本：`1.1.1`

## 功能

- `better_image_search`：从互联网搜索图片并以工具图片结果返回。
- `better_image_crop`：从历史消息中读取图片，按比例或像素裁切、放大，并保存到插件上下文。
- `better_image_transform`：翻转、旋转、等比缩放或非等比缩放上下文图片/历史消息图片，并保存到插件上下文。
- 支持在 `config.toml` 中分别启用或关闭上述三个工具。

## 配置

```toml
[plugin]
enabled = true
config_version = "1.3.0"

[tools]
search = true
crop = true
transform = true
```

## 使用说明

安装后启用插件即可使用工具。互联网搜图依赖公开搜索页面，结果可用性会受网络环境和搜索源限制。

## 更新记录

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
