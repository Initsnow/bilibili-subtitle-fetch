# Bilibili Subtitle Fetch

一个用于获取 Bilibili 视频字幕的 MCP Server，支持语言和输出格式选项。

`uv tool install --python 3.13 bilibili-subtitle-fetch`

支持以下输入：

- `BV` 号，例如 `BV1fz4y1j7Mf`
- 标准视频链接，例如 `https://www.bilibili.com/video/BV1fz4y1j7Mf?p=2`
- 短链，例如 `https://b23.tv/FAm7Xn4`

## 配置

### 凭据配置文件

首次使用时执行 `bilibili-subtitle-fetch init`，然后粘贴你的 Bilibili Cookie。

默认配置文件路径：

- Windows: `%APPDATA%\bilibili-subtitle-fetch\config.toml`
- Linux/macOS: `~/.config/bilibili-subtitle-fetch/config.toml`

也可以通过 `--config /path/to/config.toml` 指定自定义路径。

配置文件会在 `[credential]` 下保存这些字段：

- `sessdata` - 访问需要登录态的字幕接口时必需
- `bili_jct` - 自动刷新时必需
- `ac_time_value` - 自动刷新时必需
- `buvid3`、`buvid4`、`dedeuserid` - 可选，但建议一并保存

当 `sessdata`、`bili_jct` 和 `ac_time_value` 都存在时，服务会在发起需要登录态的请求前自动检查是否需要刷新 Cookie，并将新值回写到 `config.toml`，不需要手动执行刷新命令。

### 运行参数

- `--preferred-lang` - 覆盖默认字幕语言
- `--output-format` - 覆盖默认输出格式
- `--config` - 使用自定义配置文件路径

## CLI 用法

先执行 `bilibili-subtitle-fetch init` 配置 Cookie，然后可以直接在终端获取字幕：

```bash
bilibili-subtitle-fetch fetch BV1fz4y1j7Mf
bilibili-subtitle-fetch fetch "https://www.bilibili.com/video/BV1fz4y1j7Mf?p=2"
```

命令会把字幕输出到标准输出，并在成功后自动复制到剪贴板。

可选参数：

- `--preferred-lang` - 指定优先字幕语言
- `--output-format text|timestamped` - 指定输出格式
- `--no-clipboard` - 只输出，不复制到剪贴板

[获取 Bilibili 凭据](https://nemo2011.github.io/bilibili-api/#/get-credential.md)
