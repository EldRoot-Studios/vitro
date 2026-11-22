# Changelog

All notable changes to Vitro are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- A browser demo on GitHub Pages, for every system: the app with the two sample runs built in, at https://eldroot-studios.github.io/vitro/. Choose your own trace files on its start screen; they are read in the page and never uploaded. A link such as `#open=3,2&view=net&token=7` opens a given view directly. It is built by `tools/make_demo_site.py` and published by a workflow on every change to the app or the samples.
- On the start screen of the demo, links to the Windows download and to the source.
- `tools/make_screenshots.py` makes the 2:1 picture of link previews (`assets/social.png`) and can run a script in the page before a picture.

## [0.2.0] - 2026-11-22

A new look, and one program instead of two.

### Added

- **`vitro.exe`**, the whole app as one Windows program of about 6 MB. The inspector and the generation replay are six tabs of one window: Tokens, Attention, Activations, Spans, 3D Net and Replay. The sample traces and the licence notices are inside the file. There is no server and no browser, and the traces you open are read in the window and never uploaded.
- **A new interface**, a React and TypeScript app in `web/` (the built app is committed as `web/dist`). The picture fills the window and the panels lie on it as frosted glass with sharp, cut edges, on a warm grey with off-white ink and one orange accent for whatever is chosen. The data itself stays grey and white.
- A start screen with the sample runs, a file picker, files dropped on the window or on the program's icon, a map of the keys (`?`), saving the open view as a PNG (`S`), and views that survive a reload because they are kept in the address of the window (`--hash`).
- `vitro web`, the same app served to your browser on 127.0.0.1, for Linux and macOS, and `vitro app`, which opens `vitro.exe` with trace files.
- `vitro.exe --screenshot`, `--eval-file`, `--hash`, `--size`, `--wait` and `--timeout`, which drive the real window for the tests and for the pictures of the README (`tools/make_screenshots.py`).
- Tests for the new parts: the interface (32 vitest tests), the browser reader against NumPy to 1e-9 on every sample and on all 65 536 float16 patterns, the words of the refusals, the committed `web/dist` against a fresh build, the real window of `vitro.exe`, and release hygiene (one version everywhere, a changelog entry for it).

### Changed

- `vitro.exe` replaces `vitro_view.exe` and `vitro_demo.exe`. They are still built by `bash build.sh` and reachable as `vitro view` and `vitro demo`.

### Fixed

- A trace file with a damaged header, bad UTF-8 or a header that is not an object is refused with the file named and the reason, in the same words in Python, C++ and the app.
- `import vitro` no longer loads PyTorch: tools that only read saved traces start in 0.3 s instead of 6 s.

### Known limits

- `vitro.exe` is not code-signed, so Windows SmartScreen may ask you to confirm the first start.
- `vitro.exe` needs the Microsoft Edge WebView2 Runtime, which comes with Windows 11 and current Windows 10.

## [0.1.0] - 2026-11-21

First release: a visual trace inspector (`vitro_view.exe`, five views) and a step-by-step 3D generation player (`vitro_demo.exe`) for PyTorch and HuggingFace models, the `vitro-trace` Python package that records `.vtdump` and `.vtdemo` files, and two sample runs (T5-small and Qwen3-0.6B).

[Unreleased]: https://github.com/EldRoot-Studios/vitro/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/EldRoot-Studios/vitro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/EldRoot-Studios/vitro/releases/tag/v0.1.0
