"""Better Image 插件。"""

from __future__ import annotations

from base64 import b64decode, b64encode
from io import BytesIO
from maibot_sdk import MaiBotPlugin, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType
from PIL import Image
from typing import Any

import hashlib
import logging
import re


logger = logging.getLogger("plugin.better_image")

MAX_CONTEXT_IMAGES = 32
MAX_OUTPUT_EDGE = 4096
DEFAULT_OUTPUT_FORMAT = "png"


def _tool_param(
    name: str,
    param_type: ToolParamType,
    description: str,
    required: bool = True,
    default: Any = None,
    enum_values: list[Any] | None = None,
) -> ToolParameterInfo:
    """构造工具参数声明。"""

    return ToolParameterInfo(
        name=name,
        param_type=param_type,
        description=description,
        required=required,
        default=default,
        enum_values=enum_values,
    )


def _extract_nested_mapping(payload: Any) -> dict[str, Any]:
    """剥离常见的 capability 包装层。"""

    current = payload
    visited: set[int] = set()
    while isinstance(current, dict):
        current_id = id(current)
        if current_id in visited:
            break
        visited.add(current_id)

        for wrapper_key in ("result", "data"):
            nested_value = current.get(wrapper_key)
            if isinstance(nested_value, dict):
                current = nested_value
                break
        else:
            return current
    return {}


def _decode_base64_image(raw_base64: str) -> tuple[str, bytes] | None:
    """解析普通 Base64 或 data URL 图片。"""

    normalized_base64 = raw_base64.strip()
    if not normalized_base64:
        return None

    image_format = DEFAULT_OUTPUT_FORMAT
    data_url_match = re.match(
        r"^data:image/(?P<format>[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$",
        normalized_base64,
        re.DOTALL,
    )
    if data_url_match is not None:
        image_format = data_url_match.group("format").lower()
        normalized_base64 = data_url_match.group("data").strip()

    try:
        image_bytes = b64decode(normalized_base64, validate=True)
    except Exception:
        return None
    if not image_bytes:
        return None

    return image_format, image_bytes


def _iter_image_reference_values(value: Any) -> list[str]:
    """从图片消息段中收集可能的图片引用。"""

    if isinstance(value, dict):
        references: list[str] = []
        for key in ("binary_data_base64", "base64", "data_url", "data"):
            references.extend(_iter_image_reference_values(value.get(key)))
        return references

    if isinstance(value, list):
        references: list[str] = []
        for item in value:
            references.extend(_iter_image_reference_values(item))
        return references

    normalized_value = str(value or "").strip()
    if not normalized_value:
        return []
    if normalized_value.startswith("[") and normalized_value.endswith("]") and not normalized_value.startswith("[CQ:"):
        return []
    return [normalized_value]


def _normalize_output_format(output_format: str) -> str:
    """规范化输出图片格式。"""

    normalized_format = str(output_format or DEFAULT_OUTPUT_FORMAT).strip().lower()
    if normalized_format == "jpg":
        return "jpeg"
    if normalized_format in {"jpeg", "png", "webp"}:
        return normalized_format
    return DEFAULT_OUTPUT_FORMAT


def _build_crop_box(
    image_size: tuple[int, int],
    *,
    coordinate_mode: str,
    crop_x: float,
    crop_y: float,
    crop_width: float,
    crop_height: float,
) -> tuple[int, int, int, int]:
    """根据比例或像素参数构造安全裁切框。"""

    image_width, image_height = image_size
    normalized_mode = str(coordinate_mode or "ratio").strip().lower()
    if normalized_mode == "pixel":
        left = int(round(crop_x))
        top = int(round(crop_y))
        width = int(round(crop_width))
        height = int(round(crop_height))
    else:
        left = int(round(max(0.0, min(1.0, crop_x)) * image_width))
        top = int(round(max(0.0, min(1.0, crop_y)) * image_height))
        width = int(round(max(0.0, min(1.0, crop_width)) * image_width))
        height = int(round(max(0.0, min(1.0, crop_height)) * image_height))

    left = max(0, min(left, image_width - 1))
    top = max(0, min(top, image_height - 1))
    width = image_width - left if width <= 0 else width
    height = image_height - top if height <= 0 else height
    right = max(left + 1, min(image_width, left + width))
    bottom = max(top + 1, min(image_height, top + height))
    return left, top, right, bottom


def _crop_and_scale_image(
    image_bytes: bytes,
    *,
    coordinate_mode: str,
    crop_x: float,
    crop_y: float,
    crop_width: float,
    crop_height: float,
    scale: float,
    output_format: str,
) -> tuple[str, str, dict[str, int]]:
    """裁切并缩放图片，返回格式、Base64 与尺寸信息。"""

    normalized_format = _normalize_output_format(output_format)
    with Image.open(BytesIO(image_bytes)) as raw_image:
        image = raw_image.convert("RGBA")
        original_width, original_height = image.size
        crop_box = _build_crop_box(
            image.size,
            coordinate_mode=coordinate_mode,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
        )
        cropped = image.crop(crop_box)

        normalized_scale = max(0.1, min(8.0, float(scale or 1.0)))
        target_width = max(1, min(MAX_OUTPUT_EDGE, int(round(cropped.width * normalized_scale))))
        target_height = max(1, min(MAX_OUTPUT_EDGE, int(round(cropped.height * normalized_scale))))
        if (target_width, target_height) != cropped.size:
            cropped = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

        if normalized_format == "jpeg":
            output_image = cropped.convert("RGB")
            pil_format = "JPEG"
        else:
            output_image = cropped
            pil_format = normalized_format.upper()

        output_buffer = BytesIO()
        output_image.save(output_buffer, format=pil_format)
        metadata = {
            "original_width": original_width,
            "original_height": original_height,
            "crop_left": crop_box[0],
            "crop_top": crop_box[1],
            "crop_right": crop_box[2],
            "crop_bottom": crop_box[3],
            "output_width": target_width,
            "output_height": target_height,
        }
        return normalized_format, b64encode(output_buffer.getvalue()).decode("utf-8"), metadata


def _extract_message_images(message: dict[str, Any]) -> list[tuple[str, bytes]]:
    """从消息字典中提取图片。"""

    raw_message = message.get("raw_message")
    if not isinstance(raw_message, list):
        return []

    images: list[tuple[str, bytes]] = []
    for component in raw_message:
        if not isinstance(component, dict):
            continue
        if str(component.get("type") or "").strip().lower() != "image":
            continue
        for reference in _iter_image_reference_values(component):
            decoded_image = _decode_base64_image(reference)
            if decoded_image is not None:
                images.append(decoded_image)
                break
    return images


class BetterImagePlugin(MaiBotPlugin):
    """提供更细粒度图片查看和复用能力的插件。"""

    def __init__(self) -> None:
        super().__init__()
        self._context_images: dict[str, dict[str, Any]] = {}
        self._context_order: list[str] = []

    async def on_load(self) -> None:
        """插件加载完成。"""

    async def on_unload(self) -> None:
        """插件卸载前清理上下文图片。"""

        self._context_images.clear()
        self._context_order.clear()

    def _remember_context_image(self, context_key: str, payload: dict[str, Any]) -> None:
        """记录一张可供后续工具复用的上下文图片。"""

        if context_key not in self._context_images:
            self._context_order.append(context_key)
        self._context_images[context_key] = payload

        while len(self._context_order) > MAX_CONTEXT_IMAGES:
            oldest_key = self._context_order.pop(0)
            self._context_images.pop(oldest_key, None)

    async def _get_message_images(
        self,
        msg_id: str,
        *,
        stream_id: str = "",
    ) -> tuple[list[tuple[str, bytes]], str | None]:
        """读取消息中的图片，返回图片列表和错误信息。"""

        target_message_id = str(msg_id or "").strip()
        if not target_message_id:
            return [], "需要提供 msg_id。"

        lookup_result = await self.ctx.call_capability(
            "message.get_by_id",
            message_id=target_message_id,
            chat_id=str(stream_id or "").strip() or None,
            include_binary_data=True,
        )
        lookup_payload = _extract_nested_mapping(lookup_result)
        if lookup_payload and lookup_payload.get("success") is False:
            return [], str(lookup_payload.get("error") or "读取消息失败。")

        message = lookup_payload.get("message")
        if not isinstance(message, dict):
            return [], f"没有找到消息或消息格式异常：msg_id={target_message_id}"

        images = _extract_message_images(message)
        if not images:
            return [], f"目标消息中没有可读取的图片：msg_id={target_message_id}"

        return images, None

    @Tool(
        "better_image_get",
        description=(
            "获取某条历史消息中的图片，根据比例或像素参数裁切、放大，"
            "会将裁切后图片放入上下文以供后续调用,"
            "当你想展示图片中某处信息帮助你向其他人说明使使用,"
            "当图中某些信息不够清晰时使用,"
        ),
        parameters=[
            _tool_param("msg_id", ToolParamType.STRING, "包含图片的目标消息编号。"),
            _tool_param("image_index", ToolParamType.INTEGER, "同一消息中第几张图片，从 0 开始。", False, 0),
            _tool_param("coordinate_mode", ToolParamType.STRING, "裁切坐标模式：ratio 或 pixel。", False, "ratio", ["ratio", "pixel"]),
            _tool_param("crop_x", ToolParamType.FLOAT, "裁切区域左上角 x。", False, 0.0),
            _tool_param("crop_y", ToolParamType.FLOAT, "裁切区域左上角 y。", False, 0.0),
            _tool_param("crop_width", ToolParamType.FLOAT, "裁切宽度；<=0 表示到图片右边缘。", False, 1.0),
            _tool_param("crop_height", ToolParamType.FLOAT, "裁切高度；<=0 表示到图片下边缘。", False, 1.0),
            _tool_param("scale", ToolParamType.FLOAT, "放大倍率，范围会限制在 0.1 到 8.0。", False, 1.0),
            _tool_param("output_format", ToolParamType.STRING, "输出格式。", False, DEFAULT_OUTPUT_FORMAT, ["png", "jpeg", "webp"]),
            _tool_param("context_key", ToolParamType.STRING, "可选的上下文图片名称，留空则自动生成。", False, ""),
        ],
    )
    async def handle_better_image_get(
        self,
        msg_id: str = "",
        image_index: int = 0,
        coordinate_mode: str = "ratio",
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_width: float = 1.0,
        crop_height: float = 1.0,
        scale: float = 1.0,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        context_key: str = "",
        stream_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """提取、裁切并放大消息图片。"""

        del kwargs
        target_message_id = str(msg_id or "").strip()
        if not target_message_id:
            return {"success": False, "content": "better_image_get 需要提供 msg_id。"}

        images, error = await self._get_message_images(target_message_id, stream_id=stream_id)
        if error is not None:
            return {"success": False, "content": error}
        if image_index < 0 or image_index >= len(images):
            return {
                "success": False,
                "content": f"图片序号超出范围：image_index={image_index}，该消息共有 {len(images)} 张图片。",
            }

        _source_format, source_bytes = images[image_index]
        try:
            processed_format, processed_base64, metadata = _crop_and_scale_image(
                source_bytes,
                coordinate_mode=coordinate_mode,
                crop_x=float(crop_x),
                crop_y=float(crop_y),
                crop_width=float(crop_width),
                crop_height=float(crop_height),
                scale=float(scale),
                output_format=output_format,
            )
        except Exception as exc:
            logger.exception("better_image_get 图片处理失败：msg_id=%s", target_message_id)
            return {"success": False, "content": f"图片处理失败：{exc}"}

        resolved_context_key = str(context_key or "").strip()
        if not resolved_context_key:
            digest = hashlib.sha256(processed_base64.encode("utf-8")).hexdigest()[:12]
            resolved_context_key = f"{target_message_id}:{image_index}:{digest}"

        self._remember_context_image(
            resolved_context_key,
            {
                "format": processed_format,
                "base64": processed_base64,
                "message_id": target_message_id,
                "image_index": image_index,
                "metadata": metadata,
            },
        )

        content = (
            f"已获取并处理消息 {target_message_id} 的第 {image_index} 张图片，"
            f"上下文图片名称为 {resolved_context_key}，输出尺寸 {metadata['output_width']}x{metadata['output_height']}。"
        )
        output_mime_subtype = "jpeg" if processed_format == "jpeg" else processed_format

        return {
            "success": True,
            "content": content,
            "context_key": resolved_context_key,
            "image_format": processed_format,
            "image_base64": processed_base64,
            "content_items": [
                {
                    "content_type": "image",
                    "data": processed_base64,
                    "mime_type": f"image/{output_mime_subtype}",
                    "name": f"{resolved_context_key}.{processed_format}",
                    "description": content,
                    "metadata": metadata,
                }
            ],
            "metadata": metadata,
        }

    @Tool(
        "better_image_send_context",
        description=(
            "发送上下文中的图片。可以发送 better_image_get 保存的图片，也可以按 msg_id 和 index "
            "发送聊天上下文里别人发送的原图；context_key 与 msg_id 二选一。"
            "同一上下文有多张图片时用 index 指定，从 0 开始；可以连续调用多次发送多张图片。"
        ),
        parameters=[
            _tool_param("context_key", ToolParamType.STRING, "better_image_get 返回的上下文图片名称；与 msg_id 二选一。", False, ""),
            _tool_param("msg_id", ToolParamType.STRING, "包含图片的上下文消息编号；与 context_key 二选一。", False, ""),
            _tool_param("index", ToolParamType.INTEGER, "同一消息中的图片序号，从 0 开始。", False, 0),
        ],
    )
    async def handle_better_image_send_context(
        self,
        context_key: str = "",
        msg_id: str = "",
        index: int = 0,
        stream_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """发送插件上下文或消息上下文中的图片。"""

        image_index = int(kwargs.get("image_index", index) or 0)
        resolved_context_key = str(context_key or "").strip()
        target_message_id = str(msg_id or "").strip()
        if bool(resolved_context_key) == bool(target_message_id):
            return {"success": False, "content": "better_image_send_context 需要在 context_key 和 msg_id 中二选一。"}

        target_stream_id = str(stream_id or "").strip()
        if not target_stream_id:
            return {"success": False, "content": "无法确定当前聊天流，不能发送图片。"}

        source_label = resolved_context_key
        if resolved_context_key:
            context_image = self._context_images.get(resolved_context_key)
            if context_image is None:
                return {"success": False, "content": f"没有找到上下文图片：{resolved_context_key}"}
            image_base64 = str(context_image["base64"])
        else:
            images, error = await self._get_message_images(target_message_id, stream_id=target_stream_id)
            if error is not None:
                return {"success": False, "content": error}
            if image_index < 0 or image_index >= len(images):
                return {
                    "success": False,
                    "content": f"图片序号超出范围：index={image_index}，该消息共有 {len(images)} 张图片。",
                }

            image_format, image_bytes = images[image_index]
            image_base64 = b64encode(image_bytes).decode("utf-8")
            source_label = f"{target_message_id} 的第 {image_index} 张图片"

        success = await self.ctx.send.image(
            image_base64,
            target_stream_id,
            sync_to_maisaka_history=True,
            maisaka_source_kind="plugin_better_image",
        )
        if not success:
            return {"success": False, "content": f"发送上下文图片失败：{source_label}"}

        result = {
            "success": True,
            "content": f"已发送上下文图片：{source_label}",
            "stream_id": target_stream_id,
        }
        if resolved_context_key:
            result["context_key"] = resolved_context_key
        else:
            result["msg_id"] = target_message_id
            result["index"] = image_index
            result["image_format"] = image_format
        return result

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        """处理配置热重载。"""

        del scope
        del config_data
        del version


def create_plugin() -> BetterImagePlugin:
    """创建 Better Image 插件实例。"""

    return BetterImagePlugin()
