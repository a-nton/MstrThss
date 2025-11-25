"""
Plugin Discovery and Loading System

Scans modules/data_sources/ and modules/llm_configs/ directories
to find available plugins and present them to the user.
"""

import os
import importlib.util
import sys

def discover_plugins(plugin_dir):
    """
    Scans a directory for Python plugin files and extracts metadata.

    Args:
        plugin_dir: Path to directory containing plugin files

    Returns:
        list of dicts: [{"name": str, "description": str, "module_name": str, "file_path": str}, ...]
    """
    plugins = []

    if not os.path.exists(plugin_dir):
        return plugins

    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]  # Remove .py extension
            file_path = os.path.join(plugin_dir, filename)

            try:
                # Dynamically import the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Extract required metadata
                plugin_name = getattr(module, "PLUGIN_NAME", module_name)
                plugin_desc = getattr(module, "PLUGIN_DESCRIPTION", "No description")

                plugins.append({
                    "name": plugin_name,
                    "description": plugin_desc,
                    "module_name": module_name,
                    "file_path": file_path
                })

            except Exception as e:
                print(f"[WARNING] Failed to load plugin {filename}: {e}")
                continue

    return plugins

def load_plugin(plugin_dir, module_name):
    """
    Loads a specific plugin module by name.

    Args:
        plugin_dir: Directory containing the plugin
        module_name: Name of the module (without .py)

    Returns:
        module: The loaded Python module
    """
    file_path = os.path.join(plugin_dir, f"{module_name}.py")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plugin not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def select_plugin(plugins, plugin_type="plugin"):
    """
    Interactive menu for plugin selection.

    Args:
        plugins: List of plugin dicts from discover_plugins()
        plugin_type: Display name for the plugin type (e.g., "data source", "LLM config")

    Returns:
        str: Selected plugin's module_name, or None if cancelled
    """
    if not plugins:
        print(f"❌ No {plugin_type} plugins found!")
        return None

    print(f"\n{'='*50}")
    print(f"Available {plugin_type.upper()} Plugins:")
    print(f"{'='*50}")

    for idx, plugin in enumerate(plugins, 1):
        print(f"{idx}. {plugin['name']}")
        print(f"   {plugin['description']}")

    print(f"{'='*50}")

    while True:
        try:
            choice = input(f"Select {plugin_type} (1-{len(plugins)}): ").strip()
            choice_idx = int(choice) - 1

            if 0 <= choice_idx < len(plugins):
                selected = plugins[choice_idx]
                print(f"✓ Selected: {selected['name']}")
                return selected["module_name"]
            else:
                print(f"Invalid choice. Enter 1-{len(plugins)}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Selection cancelled")
            return None

if __name__ == "__main__":
    # Test the discovery system
    print("Testing plugin discovery...\n")

    data_sources = discover_plugins("modules/data_sources")
    print(f"Found {len(data_sources)} data source(s):")
    for ds in data_sources:
        print(f"  - {ds['name']}: {ds['description']}")

    print()

    llm_configs = discover_plugins("modules/llm_configs")
    print(f"Found {len(llm_configs)} LLM config(s):")
    for lc in llm_configs:
        print(f"  - {lc['name']}: {lc['description']}")
