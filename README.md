# Better Image

Better Image 是一个 MaiBot 插件，提供互联网搜图、历史消息图片提取、裁切放大和上下文图片发送能力。

## 功能

- `better_image_search`：从互联网搜索图片并以工具图片结果返回。
- `better_image_get`：从历史消息中读取图片，按比例或像素裁切、放大，并保存到插件上下文。
- `better_image_send_context`：发送插件上下文中的图片，或按消息 ID 和序号发送聊天上下文里的原图。
- 支持在 `config.toml` 中分别启用或关闭上述三个工具。

## 配置

```toml
[plugin]
enabled = true
config_version = "1.1.0"

[tools]
search = true
get = true
send_context = true
```

## 使用说明

安装后启用插件即可使用工具。互联网搜图依赖公开搜索页面，结果可用性会受网络环境和搜索源限制。

## 许可证

MIT
