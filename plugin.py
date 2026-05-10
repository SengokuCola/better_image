"""Better Image 插件。"""

from __future__ import annotations

from base64 import b64decode, b64encode
from io import BytesIO
from typing import Any, Mapping
from urllib.parse import quote_plus, urlparse

from maibot_sdk import MaiBotPlugin, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType
from PIL import Image

import hashlib
import httpx
import json
import logging
import re


logger = logging.getLogger("plugin.better_image")

MAX_CONTEXT_IMAGES = 32
MAX_OUTPUT_EDGE = 4096
DEFAULT_OUTPUT_FORMAT = "png"
PLUGIN_CONFIG_VERSION = "1.1.0"
DEFAULT_SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
MAX_SEARCH_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SEARCH_RESULT_IMAGES = 5

DEFAULT_PLUGIN_CONFIG: dict[str, Any] = {
    "plugin": {
        "enabled": True,
        "config_version": PLUGIN_CONFIG_VERSION,
    },
    "tools": {
        "search": True,
        "get": True,
        "send_context": True,
    },
}


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


def _normalize_downloaded_image(image_bytes: bytes) -> tuple[str, str, dict[str, int]] | None:
    """校验下载图片并转成工具可稳定消费的 Base64。"""

    try:
        with Image.open(BytesIO(image_bytes)) as raw_image:
            raw_image.verify()
        with Image.open(BytesIO(image_bytes)) as raw_image:
            image = raw_image.convert("RGBA")
            output_buffer = BytesIO()
            image.save(output_buffer, format="PNG")
            return (
                DEFAULT_OUTPUT_FORMAT,
                b64encode(output_buffer.getvalue()).decode("utf-8"),
                {
                    "original_width": raw_image.width,
                    "original_height": raw_image.height,
                    "output_width": image.width,
                    "output_height": image.height,
                },
            )
    except Exception:
        return None


def _deduplicate_urls(urls: list[str]) -> list[str]:
    """按顺序去重并过滤明显不可下载的图片地址。"""

    deduplicated_urls: list[str] = []
    seen_urls: set[str] = set()
    for raw_url in urls:
        image_url = str(raw_url or "").strip()
        parsed_url = urlparse(image_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            continue
        if image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        deduplicated_urls.append(image_url)
    return deduplicated_urls


def _extract_bing_image_urls(html: str) -> list[str]:
    """从 Bing 图片搜索结果页中提取原图地址。"""

    urls: list[str] = []
    for metadata_match in re.finditer(r"m=(\{.+?\})", html):
        raw_metadata = metadata_match.group(1)
        try:
            metadata = json.loads(raw_metadata.replace("&quot;", '"'))
        except Exception:
            continue
        image_url = metadata.get("murl")
        if isinstance(image_url, str):
            urls.append(image_url)

    urls.extend(match.group(1) for match in re.finditer(r'"murl"\s*:\s*"([^"]+)"', html))
    return _deduplicate_urls(urls)


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


def _deep_copy_value(value: Any) -> Any:
    """复制插件配置中的简单结构。"""

    if isinstance(value, dict):
        return _deep_copy_mapping(value)
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    return value


def _deep_copy_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """复制配置字典，避免共享默认配置对象。"""

    return {str(key): _deep_copy_value(value) for key, value in mapping.items()}


def _merge_mapping(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """把用户配置覆盖到默认配置上。"""

    for key, value in source.items():
        target_key = str(key)
        target_value = target.get(target_key)
        if isinstance(target_value, dict) and isinstance(value, Mapping):
            _merge_mapping(target_value, value)
        else:
            target[target_key] = _deep_copy_value(value)


def _ensure_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """确保配置中的某一段是字典。"""

    value = config.get(key)
    if isinstance(value, dict):
        return value
    section: dict[str, Any] = {}
    config[key] = section
    return section


class BetterImagePlugin(MaiBotPlugin):
    """提供更细粒度图片查看和复用能力的插件。"""

    def __init__(self) -> None:
        super().__init__()
        self._context_images: dict[str, dict[str, Any]] = {}
        self._context_order: list[str] = []
        self._plugin_config = _deep_copy_mapping(DEFAULT_PLUGIN_CONFIG)

    async def on_load(self) -> None:
        """插件加载完成。"""

    async def on_unload(self) -> None:
        """插件卸载前清理上下文图片。"""

        self._context_images.clear()
        self._context_order.clear()

    def get_default_config(self) -> dict[str, Any]:
        """返回插件默认配置。"""

        return _deep_copy_mapping(DEFAULT_PLUGIN_CONFIG)

    def normalize_plugin_config(self, config_data: Mapping[str, Any] | None) -> tuple[dict[str, Any], bool]:
        """补齐并规范化插件配置。"""

        normalized_config = self.get_default_config()
        current_config = _deep_copy_mapping(config_data or {})
        _merge_mapping(normalized_config, current_config)

        plugin_section = _ensure_mapping(normalized_config, "plugin")
        plugin_section["enabled"] = bool(plugin_section.get("enabled", True))
        plugin_section["config_version"] = PLUGIN_CONFIG_VERSION

        tools_section = _ensure_mapping(normalized_config, "tools")
        for tool_name in ("search", "get", "send_context"):
            tools_section[tool_name] = bool(tools_section.get(tool_name, True))

        return normalized_config, normalized_config != current_config

    def set_plugin_config(self, config: dict[str, Any]) -> None:
        """接收运行时注入的插件配置。"""

        normalized_config, _changed = self.normalize_plugin_config(config)
        self._plugin_config = normalized_config

    def get_webui_config_schema(
        self,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> dict[str, Any]:
        """返回 WebUI 配置 Schema。"""

        return {
            "plugin_id": plugin_id,
            "plugin_info": {
                "name": plugin_name or "Better Image",
                "version": plugin_version,
                "description": plugin_description,
                "author": plugin_author,
            },
            "sections": {
                "plugin": {
                    "title": "插件",
                    "fields": {
                        "enabled": {
                            "type": "boolean",
                            "label": "启用插件",
                            "default": True,
                            "ui_type": "switch",
                        }
                    },
                },
                "tools": {
                    "title": "工具开关",
                    "fields": {
                        "search": {
                            "type": "boolean",
                            "label": "启用互联网搜图工具",
                            "default": True,
                            "ui_type": "switch",
                        },
                        "get": {
                            "type": "boolean",
                            "label": "启用历史图片提取工具",
                            "default": True,
                            "ui_type": "switch",
                        },
                        "send_context": {
                            "type": "boolean",
                            "label": "启用上下文图片发送工具",
                            "default": True,
                            "ui_type": "switch",
                        },
                    },
                },
            },
            "layout": {"type": "auto", "tabs": []},
        }

    def _is_tool_enabled(self, tool_name: str) -> bool:
        """检查插件与指定工具是否启用。"""

        plugin_section = self._plugin_config.get("plugin")
        tools_section = self._plugin_config.get("tools")
        if not isinstance(plugin_section, dict) or not bool(plugin_section.get("enabled", True)):
            return False
        if not isinstance(tools_section, dict):
            return True
        return bool(tools_section.get(tool_name, True))

    def _disabled_tool_result(self, tool_label: str) -> dict[str, Any]:
        """构造工具关闭时的返回结果。"""

        return {"success": False, "content": f"{tool_label} 已在 better_image 配置中关闭。"}

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

    async def _search_duckduckgo_image_urls(
        self,
        client: httpx.AsyncClient,
        query: str,
        *,
        safe_search: str,
        candidate_limit: int,
    ) -> list[str]:
        """通过 DuckDuckGo 图片搜索收集候选图片地址。"""

        search_url = "https://duckduckgo.com/"
        search_response = await client.get(
            search_url,
            params={"q": query, "iax": "images", "ia": "images"},
        )
        search_response.raise_for_status()
        token_match = re.search(r"vqd=['\"](?P<token>[^'\"]+)['\"]", search_response.text)
        if token_match is None:
            return []

        safe_search_value = {"off": "-1", "moderate": "1", "strict": "1"}.get(safe_search, "1")
        image_response = await client.get(
            "https://duckduckgo.com/i.js",
            params={
                "l": "wt-wt",
                "o": "json",
                "q": query,
                "vqd": token_match.group("token"),
                "f": ",,,",
                "p": safe_search_value,
            },
            headers={"Referer": str(search_response.url)},
        )
        image_response.raise_for_status()
        payload = image_response.json()
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        image_urls: list[str] = []
        for result in results[:candidate_limit]:
            if not isinstance(result, dict):
                continue
            image_url = result.get("image")
            if isinstance(image_url, str):
                image_urls.append(image_url)
        return _deduplicate_urls(image_urls)

    async def _search_bing_image_urls(
        self,
        client: httpx.AsyncClient,
        query: str,
        *,
        safe_search: str,
        candidate_limit: int,
    ) -> list[str]:
        """通过 Bing 图片搜索页收集候选图片地址。"""

        safe_search_value = {"off": "off", "moderate": "moderate", "strict": "strict"}.get(safe_search, "moderate")
        response = await client.get(
            f"https://www.bing.com/images/search?q={quote_plus(query)}&safeSearch={safe_search_value}&form=HDRSC2",
        )
        response.raise_for_status()
        return _extract_bing_image_urls(response.text)[:candidate_limit]

    async def _search_image_urls(
        self,
        client: httpx.AsyncClient,
        query: str,
        *,
        safe_search: str,
        candidate_limit: int,
    ) -> list[str]:
        """从多个搜索来源收集候选图片地址。"""

        for search_func in (self._search_duckduckgo_image_urls, self._search_bing_image_urls):
            try:
                image_urls = await search_func(
                    client,
                    query,
                    safe_search=safe_search,
                    candidate_limit=candidate_limit,
                )
            except Exception as exc:
                logger.warning("better_image_search 搜索来源失败：%s", exc)
                continue
            if image_urls:
                return image_urls
        return []

    async def _download_search_image(
        self,
        client: httpx.AsyncClient,
        image_url: str,
    ) -> tuple[str, str, dict[str, int]] | None:
        """下载并规范化搜索结果图片。"""

        try:
            async with client.stream("GET", image_url) as response:
                response.raise_for_status()
                content_type = str(response.headers.get("content-type") or "").lower()
                if content_type and "image/" not in content_type:
                    return None

                image_buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    image_buffer.extend(chunk)
                    if len(image_buffer) > MAX_SEARCH_IMAGE_BYTES:
                        return None
        except Exception as exc:
            logger.debug("better_image_search 下载候选图片失败：url=%s, error=%s", image_url, exc)
            return None

        return _normalize_downloaded_image(bytes(image_buffer))

    @Tool(
        "better_image_search",
        description=(
            "从互联网搜索图片并返回图片结果。适合需要查看网络上的图片、表情包、人物、物品或场景时使用；"
            "返回的图片会加入插件上下文，之后可以用 better_image_send_context 按 context_key 发送。"
        ),
        parameters=[
            _tool_param("query", ToolParamType.STRING, "图片搜索关键词。"),
            _tool_param("limit", ToolParamType.INTEGER, "返回图片数量，范围 1 到 5。", False, 3),
            _tool_param("safe_search", ToolParamType.STRING, "安全搜索级别。", False, "moderate", ["off", "moderate", "strict"]),
            _tool_param("context_prefix", ToolParamType.STRING, "可选的上下文图片名称前缀，留空则自动生成。", False, ""),
        ],
    )
    async def handle_better_image_search(
        self,
        query: str = "",
        limit: int = 3,
        safe_search: str = "moderate",
        context_prefix: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """搜索互联网图片并作为工具图片结果返回。"""

        del kwargs
        if not self._is_tool_enabled("search"):
            return self._disabled_tool_result("互联网搜图工具")

        normalized_query = str(query or "").strip()
        if not normalized_query:
            return {"success": False, "content": "better_image_search 需要提供 query。"}

        normalized_limit = max(1, min(MAX_SEARCH_RESULT_IMAGES, int(limit or 1)))
        normalized_safe_search = str(safe_search or "moderate").strip().lower()
        if normalized_safe_search not in {"off", "moderate", "strict"}:
            normalized_safe_search = "moderate"

        headers = {"User-Agent": DEFAULT_SEARCH_USER_AGENT, "Accept": "text/html,application/json,image/*,*/*"}
        timeout = httpx.Timeout(15.0, connect=8.0)
        candidate_limit = max(12, normalized_limit * 6)
        async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
            image_urls = await self._search_image_urls(
                client,
                normalized_query,
                safe_search=normalized_safe_search,
                candidate_limit=candidate_limit,
            )
            if not image_urls:
                return {"success": False, "content": f"没有搜索到可用图片：{normalized_query}"}

            results: list[dict[str, Any]] = []
            content_items: list[dict[str, Any]] = []
            context_prefix_value = str(context_prefix or "").strip()
            if not context_prefix_value:
                query_digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:8]
                context_prefix_value = f"search:{query_digest}"

            for image_url in image_urls:
                if len(results) >= normalized_limit:
                    break

                downloaded_image = await self._download_search_image(client, image_url)
                if downloaded_image is None:
                    continue

                image_format, image_base64, metadata = downloaded_image
                digest = hashlib.sha256(image_base64.encode("utf-8")).hexdigest()[:12]
                context_key = f"{context_prefix_value}:{len(results)}:{digest}"
                self._remember_context_image(
                    context_key,
                    {
                        "format": image_format,
                        "base64": image_base64,
                        "source": "internet_search",
                        "query": normalized_query,
                        "url": image_url,
                        "metadata": metadata,
                    },
                )

                description = f"搜索“{normalized_query}”得到的第 {len(results)} 张图片，来源：{image_url}"
                content_items.append(
                    {
                        "content_type": "image",
                        "data": image_base64,
                        "mime_type": f"image/{image_format}",
                        "name": f"{context_key}.{image_format}",
                        "description": description,
                        "metadata": metadata | {"source_url": image_url, "context_key": context_key},
                    }
                )
                results.append(
                    {
                        "context_key": context_key,
                        "image_format": image_format,
                        "source_url": image_url,
                        "metadata": metadata,
                    }
                )

        if not results:
            return {"success": False, "content": f"搜索到了候选图片，但都无法下载或解析：{normalized_query}"}

        content = (
            f"已搜索“{normalized_query}”并返回 {len(results)} 张图片。"
            "如需发送其中某张图片，请调用 better_image_send_context 并传入对应 context_key。"
        )
        return {
            "success": True,
            "content": content,
            "query": normalized_query,
            "results": results,
            "context_keys": [result["context_key"] for result in results],
            "content_items": content_items,
        }

    @Tool(
        "better_image_get",
        description=(
            "获取某条历史消息中的图片，根据比例或像素参数裁切、放大，"
            "会将裁切后图片放入上下文以供后续调用,"
            "当你想展示图片中某处信息帮助你向其他人说明使使用,"
            "当图中某些信息不够清晰时使用。"
            "！注意，截图完成后请你检查返回的图片是否包含了你想要的信息，并且没有被裁切掉重要部分，必要时可以调整裁切参数重新获取。"
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
        if not self._is_tool_enabled("get"):
            return self._disabled_tool_result("历史图片提取工具")

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
            f"请你检查返回的图片是否包含了你想要的信息，并且没有被裁切掉重要部分，必要时可以调整裁切参数重新获取。"
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
                    "metadata": metadata | {"context_key": resolved_context_key},
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

        if not self._is_tool_enabled("send_context"):
            return self._disabled_tool_result("上下文图片发送工具")

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
        del version
        self.set_plugin_config(config_data)


def create_plugin() -> BetterImagePlugin:
    """创建 Better Image 插件实例。"""

    return BetterImagePlugin()
