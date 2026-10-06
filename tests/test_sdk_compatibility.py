from base64 import b64decode, b64encode
from io import BytesIO

from maibot_sdk.context import PluginContext
from PIL import Image

import asyncio


def test_crop_uses_sdk_message_query_and_returns_image(plugin_module):
    source = BytesIO()
    Image.new("RGB", (120, 80), "red").save(source, format="PNG")

    async def rpc_call(method, plugin_id, payload):
        assert method == "cap.call"
        assert payload == {
            "capability": "message.get_by_id",
            "args": {
                "message_id": "message-1",
                "chat_id": "session-1",
                "stream_id": "",
                "include_binary_data": True,
            },
        }
        return {
            "success": True,
            "message": {
                "raw_message": [{
                    "type": "image",
                    "data": "[图片]",
                    "binary_data_base64": b64encode(source.getvalue()).decode(),
                }],
            },
        }

    plugin = plugin_module.create_plugin()
    plugin._set_context(PluginContext("better-image", rpc_call=rpc_call))
    result = asyncio.run(plugin.handle_image_crop(msg_id="message-1", stream_id="session-1", crop_width=0.5, scale=2))
    assert result["success"]
    with Image.open(BytesIO(b64decode(result["content_items"][0]["data"]))) as image:
        assert image.size == (120, 160)


def test_message_query_failure_is_reported(plugin_module):
    async def rpc_call(method, plugin_id, payload):
        return {"success": False, "error": "database unavailable"}

    plugin = plugin_module.create_plugin()
    plugin._set_context(PluginContext("better-image", rpc_call=rpc_call))
    result = asyncio.run(plugin.handle_image_crop(msg_id="message-1"))
    assert not result["success"]
    assert "database unavailable" in result["content"]
