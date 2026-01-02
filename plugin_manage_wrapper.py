"""
Wrapper around the host application's plugin management API.
Provides a small, stable interface the MaiVecMem plugin can use for managing plugins.

Functions:
- list_loaded_plugins() -> List[str]
- list_registered_plugins() -> List[str]
- get_plugin_path(plugin_name: str) -> str
- remove_plugin(plugin_name: str) -> bool
- reload_plugin(plugin_name: str) -> bool
- load_plugin(plugin_name: str) -> Tuple[bool, int]
- add_plugin_directory(plugin_directory: str) -> bool
- rescan_plugin_directory() -> Tuple[int, int]

This wrapper will try multiple import paths for compatibility with host.
"""

from typing import List, Tuple
import os
import sys

# Try import paths
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.abspath(os.path.join(current_dir, "../.."))
    if target_path not in sys.path:
        sys.path.insert(0, target_path)
    from src.plugin_system.apis import plugin_manage_api
except Exception:
    try:
        from src.plugin_system import plugin_manage_api
    except Exception:
        plugin_manage_api = None


def _ensure_api():
    if plugin_manage_api is None:
        raise RuntimeError("plugin_manage_api not available in host environment")


def list_loaded_plugins() -> List[str]:
    _ensure_api()
    try:
        return plugin_manage_api.list_loaded_plugins()
    except Exception:
        # fallback: try attribute
        return plugin_manage_api.list_loaded_plugins()


def list_registered_plugins() -> List[str]:
    _ensure_api()
    try:
        return plugin_manage_api.list_registered_plugins()
    except Exception:
        return plugin_manage_api.list_registered_plugins()


def get_plugin_path(plugin_name: str) -> str:
    _ensure_api()
    try:
        path = plugin_manage_api.get_plugin_path(plugin_name)
    except Exception:
        path = plugin_manage_api.get_plugin_path(plugin_name)
    if not path:
        raise ValueError(f"Plugin {plugin_name} not found")
    return path


async def remove_plugin(plugin_name: str) -> bool:
    _ensure_api()
    try:
        return await plugin_manage_api.remove_plugin(plugin_name)
    except Exception:
        return await plugin_manage_api.remove_plugin(plugin_name)


async def reload_plugin(plugin_name: str) -> bool:
    _ensure_api()
    try:
        return await plugin_manage_api.reload_plugin(plugin_name)
    except Exception:
        return await plugin_manage_api.reload_plugin(plugin_name)


def load_plugin(plugin_name: str) -> Tuple[bool, int]:
    _ensure_api()
    try:
        return plugin_manage_api.load_plugin(plugin_name)
    except Exception:
        return plugin_manage_api.load_plugin(plugin_name)


def add_plugin_directory(plugin_directory: str) -> bool:
    _ensure_api()
    try:
        return plugin_manage_api.add_plugin_directory(plugin_directory)
    except Exception:
        return plugin_manage_api.add_plugin_directory(plugin_directory)


def rescan_plugin_directory() -> Tuple[int, int]:
    _ensure_api()
    try:
        return plugin_manage_api.rescan_plugin_directory()
    except Exception:
        return plugin_manage_api.rescan_plugin_directory()
