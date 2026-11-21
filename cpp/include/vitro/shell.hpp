// Win32 HWND host shell shared by Vitro viewer/player and downstream tools:
// menus, Open/Save dialogs, drag-and-drop, DPI scaling, fullscreen, 60 Hz animation loop, and headless event playback.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#include <mmsystem.h>
#include <shellapi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

#include "canvas.hpp"

namespace vitro {

struct MenuItem {
    int id = 0;                 // 0: separator
    std::string text;           // UTF-8; optional '\t' before shortcut
    bool checked = false;
};
struct Menu {
    std::string title;
    std::vector<MenuItem> items;
};

enum ShellCommand { CMD_OPEN = 100, CMD_SAVE_PNG, CMD_EXIT, CMD_ZOOM_IN, CMD_ZOOM_OUT, CMD_ZOOM_RESET, CMD_FULLSCREEN, CMD_APP_FIRST = 1000 };

class ShellApp {
public:
    virtual ~ShellApp() = default;
    virtual Canvas& canvas() = 0;
    virtual void render() = 0;
    virtual void move(int x, int y) = 0;
    virtual void press(int x, int y) = 0;
    virtual void release(int x, int y) = 0;
    virtual void wheel(int x, int y, int delta, bool ctrl) = 0;
    virtual bool key(int vk, bool ctrl, bool shift) = 0;
    virtual bool tick(double seconds) = 0;
    virtual bool escape() { return false; }
    virtual int take_request() { return 0; }
    virtual void notice(const std::string& text) { (void)text; }
    virtual std::string open_file(const std::string& path) = 0;
    virtual std::string title() = 0;
    virtual float ui_zoom() const = 0;
    virtual void set_ui_zoom(float zoom) = 0;
    virtual std::vector<Menu> menus() = 0;
    virtual void command(int id) = 0;
    virtual std::string state_json() = 0;
    virtual std::string status_line() const = 0;
};

struct LibraryItem {
    std::string path, name;
    unsigned long long bytes = 0;
    long long mtime = 0;
};

inline std::wstring widen(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), &w[0], n);
    return w;
}
inline std::string narrow(const std::wstring& w) {
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), &s[0], n, nullptr, nullptr);
    return s;
}

inline std::string json_escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", c);
                    o += b;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

inline std::string exe_dir() {
    wchar_t buf[MAX_PATH * 2];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 2);
    std::string p = narrow(std::wstring(buf, n));
    size_t cut = p.find_last_of("/\\");
    return cut == std::string::npos ? "." : p.substr(0, cut);
}

inline bool dir_exists(const std::string& d) {
    DWORD a = GetFileAttributesW(widen(d).c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

inline float display_scale() {
    SetProcessDPIAware();
    HDC dc = GetDC(nullptr);
    float s = GetDeviceCaps(dc, LOGPIXELSX) / 96.0f;
    ReleaseDC(nullptr, dc);
    return s;
}

inline std::string file_name_of(const std::string& path) {
    size_t cut = path.find_last_of("/\\");
    return cut == std::string::npos ? path : path.substr(cut + 1);
}

inline std::vector<std::string> read_recent(const std::string& file) {
    std::vector<std::string> out;
    FILE* fh = _wfopen(widen(file).c_str(), L"rb");
    if (!fh) return out;
    char line[2048];
    while (std::fgets(line, sizeof line, fh) && out.size() < 12) {
        std::string p = line;
        while (!p.empty() && (p.back() == '\n' || p.back() == '\r')) p.pop_back();
        if (!p.empty()) out.push_back(p);
    }
    std::fclose(fh);
    return out;
}

inline bool write_recent(const std::string& file, const std::vector<std::string>& list) {
    if (file.empty()) return true;
    std::string tmp = file + ".part";
    FILE* fh = _wfopen(widen(tmp).c_str(), L"wb");
    if (!fh) return false;
    for (const std::string& p : list) std::fprintf(fh, "%s\n", p.c_str());
    bool ok = std::fflush(fh) == 0;
    std::fclose(fh);
    return ok && MoveFileExW(widen(tmp).c_str(), widen(file).c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
}

inline bool stat_file(const std::string& path, LibraryItem* out) {
    WIN32_FILE_ATTRIBUTE_DATA d;
    if (!GetFileAttributesExW(widen(path).c_str(), GetFileExInfoStandard, &d)) return false;
    out->path = path;
    out->name = file_name_of(path);
    out->bytes = (static_cast<unsigned long long>(d.nFileSizeHigh) << 32) | d.nFileSizeLow;
    out->mtime = ((static_cast<long long>(d.ftLastWriteTime.dwHighDateTime) << 32) | d.ftLastWriteTime.dwLowDateTime) / 10000000LL - 11644473600LL;
    return true;
}

inline std::vector<LibraryItem> scan_library(const std::string& dir_in, const std::string& ext, const std::vector<std::string>& recent) {
    std::vector<LibraryItem> items;
    auto add = [&](const LibraryItem& it) {
        for (const LibraryItem& have : items)
            if (have.path == it.path) return;
        items.push_back(it);
    };
    for (const std::string& p : recent) {
        LibraryItem it;
        if (stat_file(p, &it)) add(it);
    }
    std::string dir = dir_in;
    if (dir.empty() && !recent.empty()) dir = recent.front().substr(0, recent.front().find_last_of("/\\"));
    if (!dir.empty()) {
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW(widen(dir + "/*." + ext).c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                LibraryItem it;
                if (stat_file(dir + "/" + narrow(fd.cFileName), &it)) add(it);
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
    }
    std::stable_sort(items.begin(), items.end(), [](const LibraryItem& a, const LibraryItem& b) { return a.mtime > b.mtime; });
    if (items.size() > 40) items.resize(40);
    return items;
}

struct Event {
    char kind = 0;              // m move, p press, r release, k key, K Ctrl+key, w wheel, o open, t time
    int a = 0, b = 0, c = 0;
    std::string text;
};

inline std::vector<Event> parse_events(const std::string& text) {
    std::vector<Event> out;
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find(';', at);
        if (end == std::string::npos) end = text.size();
        std::string item = text.substr(at, end - at);
        at = end + 1;
        if (item.empty()) continue;
        Event e;
        e.kind = item[0];
        if (e.kind == 'o') {
            e.text = item.substr(1);
            if (e.text.empty()) throw std::runtime_error("--events: 'o' needs a file name");
        } else if (e.kind == 't') {
            e.text = item.substr(1);
            char* stop = nullptr;
            std::strtod(e.text.c_str(), &stop);
            if (e.text.empty() || *stop) throw std::runtime_error("--events: cannot read '" + item + "' (t seconds)");
        } else {
            int n = std::sscanf(item.c_str() + 1, "%d,%d,%d", &e.a, &e.b, &e.c);
            bool ok = ((e.kind == 'k' || e.kind == 'K') && n >= 1) || ((e.kind == 'm' || e.kind == 'p' || e.kind == 'r') && n >= 2) || (e.kind == 'w' && n == 3);
            if (!ok) throw std::runtime_error("--events: cannot read '" + item + "' (m x,y | p x,y | r x,y | k code | K code | w x,y,delta | t seconds | o file)");
        }
        out.push_back(e);
    }
    return out;
}

inline std::string play_events(ShellApp& app, const std::vector<Event>& events, std::vector<int>* requests = nullptr) {
    std::string errors;
    for (const Event& e : events) {
        app.render();
        switch (e.kind) {
            case 'm': app.move(e.a, e.b); break;
            case 'p': app.press(e.a, e.b); break;
            case 'r': app.release(e.a, e.b); break;
            case 'k':
                if (e.a == VK_ESCAPE) app.escape();
                else app.key(e.a, false, false);
                break;
            case 'K':
                if (e.a == 0xBB || e.a == 0x6B) app.set_ui_zoom(app.ui_zoom() * 1.1f);
                else if (e.a == 0xBD || e.a == 0x6D) app.set_ui_zoom(app.ui_zoom() / 1.1f);
                else if (e.a == '0' || e.a == 0x60) app.set_ui_zoom(1.0f);
                else if (e.a == 'O' || e.a == 'S') {
                    if (requests) requests->push_back(e.a == 'O' ? CMD_OPEN : CMD_SAVE_PNG);
                } else app.key(e.a, true, false);
                break;
            case 'w': app.move(e.a, e.b); app.wheel(e.a, e.b, e.c, false); break;
            case 't': app.tick(std::atof(e.text.c_str())); break;
            case 'o': {
                std::string why = app.open_file(e.text);
                if (!why.empty()) errors += why + "\n";
                break;
            }
            default: break;
        }
        for (int id = app.take_request(); id != 0; id = app.take_request())
            if (requests) requests->push_back(id);
    }
    return errors;
}

inline void play_events_report(ShellApp& app, const std::string& text) {
    std::vector<int> requests;
    std::string errors = play_events(app, parse_events(text), &requests);
    if (!errors.empty()) std::fprintf(stderr, "events: %s", errors.c_str());
    if (!requests.empty()) {
        std::printf("requests:");
        for (int id : requests) std::printf(" %d", id);
        std::printf("\n");
    }
}

struct ShellConfig {
    std::wstring window_class = L"VitroShell";
    std::string extension = "vtdump";
    std::string filter_name = "Trace Files";
    std::string all_files_filter = "All Files (*.*)";
    std::string png_filter = "PNG Image (*.png)";
    std::string open_error_title = "Failed to open file";
    std::string save_error_title = "Failed to save image";
    std::string saved_notice_prefix = "saved: ";
    std::string shots_dir = ".";
    std::string app_name = "vitro";
    int width = 1500, height = 940;
    int min_width = 0, min_height = 0;
    bool quit_on_escape = true;
    bool window_test = false;
    std::string first_file;
    std::string test_drop_file;
    bool test_keep_size = false;
    int test_flood = 0;
};

namespace shell_detail {

struct State {
    ShellApp* app = nullptr;
    ShellConfig cfg;
    HWND hwnd = nullptr;
    HMENU bar = nullptr;
    std::string menu_sig, shown_title;
    std::string trace;
    double last_tick_s = 0;
    bool dirty = true;
    bool shown_stale = false;
    int frames = 0, shown = 0;
    double render_ms = 0, blit_ms = 0;
    bool failed = false;
    bool full = false;
    LONG style = 0;
    WINDOWPLACEMENT placement = {};
    std::string last_error;
};
inline State* g = nullptr;

inline std::string pick_open(HWND hwnd, const ShellConfig& cfg) {
    wchar_t file[MAX_PATH * 4] = L"";
    std::wstring filter = widen(cfg.filter_name + " (*." + cfg.extension + ")") + L'\0' + widen("*." + cfg.extension) + L'\0' + widen(cfg.all_files_filter) + L'\0' + L"*.*" + L'\0';
    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof ofn;
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = filter.c_str();
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH * 4;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER;
    return GetOpenFileNameW(&ofn) ? narrow(file) : std::string();
}

inline std::string pick_save(HWND hwnd, const ShellConfig& cfg, const std::string& name) {
    wchar_t file[MAX_PATH * 4];
    std::wstring w = widen(name);
    wcsncpy(file, w.c_str(), MAX_PATH * 4 - 1);
    file[MAX_PATH * 4 - 1] = 0;
    std::wstring filter = widen(cfg.png_filter) + L'\0' + L"*.png" + L'\0';
    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof ofn;
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = filter.c_str();
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH * 4;
    ofn.lpstrDefExt = L"png";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_EXPLORER;
    return GetSaveFileNameW(&ofn) ? narrow(file) : std::string();
}

inline double now_s() {
    static const double per_tick = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return 1.0 / static_cast<double>(f.QuadPart);
    }();
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return static_cast<double>(t.QuadPart) * per_tick;
}

inline void render_now() {
    const double t0 = now_s();
    g->app->render();
    g->render_ms += (now_s() - t0) * 1000.0;
    ++g->frames;
    g->dirty = false;
    g->shown_stale = true;
}

inline void blit_now() {
    const double t0 = now_s();
    Canvas& c = g->app->canvas();
    BITMAPINFO bi = {};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = c.W;
    bi.bmiHeader.biHeight = -c.H;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    HDC dc = GetDC(g->hwnd);
    SetDIBitsToDevice(dc, 0, 0, static_cast<DWORD>(c.W), static_cast<DWORD>(c.H), 0, 0, 0, static_cast<UINT>(c.H), c.px.data(), &bi, DIB_RGB_COLORS);
    ReleaseDC(g->hwnd, dc);
    g->blit_ms += (now_s() - t0) * 1000.0;
    ++g->shown;
    g->shown_stale = false;
}

inline void paint() {
    if (g->dirty) render_now();
    blit_now();
}

inline void ensure_fresh() {
    if (g->dirty) render_now();
}

inline std::string menu_signature(const std::vector<Menu>& ms) {
    std::string s;
    for (const Menu& m : ms) {
        s += m.title; s += '\x01';
        for (const MenuItem& it : m.items) s += std::to_string(it.id) + ":" + it.text + (it.checked ? "+" : "-") + '\x02';
        s += '\x03';
    }
    return s;
}

inline void refresh_menu() {
    std::vector<Menu> ms = g->app->menus();
    std::string sig = menu_signature(ms);
    if (g->bar && sig == g->menu_sig) return;
    HMENU fresh = CreateMenu();
    for (const Menu& m : ms) {
        HMENU pop = CreatePopupMenu();
        for (const MenuItem& it : m.items) {
            if (it.id == 0) AppendMenuW(pop, MF_SEPARATOR, 0, nullptr);
            else AppendMenuW(pop, MF_STRING | (it.checked ? MF_CHECKED : 0), static_cast<UINT_PTR>(it.id), widen(it.text).c_str());
        }
        AppendMenuW(fresh, MF_POPUP, reinterpret_cast<UINT_PTR>(pop), widen(m.title).c_str());
    }
    HMENU old = g->bar;
    g->bar = fresh;
    g->menu_sig = sig;
    if (!g->full) { SetMenu(g->hwnd, fresh); DrawMenuBar(g->hwnd); }
    if (old) DestroyMenu(old);
}

inline void refresh_title() {
    std::string t = g->app->title();
    if (t == g->shown_title) return;
    SetWindowTextW(g->hwnd, widen(t).c_str());
    g->shown_title = t;
}

inline void refresh_chrome() { refresh_menu(); refresh_title(); }
inline void redraw() { g->dirty = true; }

inline void open_path(const std::string& path) {
    std::string why = g->app->open_file(path);
    refresh_chrome();
    redraw();
    if (!why.empty()) {
        g->last_error = why;
        if (!g->cfg.window_test) MessageBoxW(g->hwnd, widen(why).c_str(), widen(g->cfg.open_error_title).c_str(), MB_OK | MB_ICONWARNING);
    }
}

inline void screenshot(const std::string& chosen) {
    ensure_fresh();
    Canvas& c = g->app->canvas();
    std::string path = chosen;
    if (path.empty()) {
        std::string dir = g->cfg.shots_dir;
        CreateDirectoryW(widen(dir).c_str(), nullptr);
        std::time_t t = std::time(nullptr);
        char stamp[32];
        std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", std::localtime(&t));
        path = dir + "/" + g->cfg.app_name + "_" + stamp + ".png";
    }
    try {
        c.save_png(path);
    } catch (const std::exception& e) {
        g->last_error = e.what();
        if (!g->cfg.window_test) MessageBoxW(g->hwnd, widen(e.what()).c_str(), widen(g->cfg.save_error_title).c_str(), MB_OK | MB_ICONWARNING);
        return;
    }
    g->app->notice(g->cfg.saved_notice_prefix + path);
}

inline void toggle_fullscreen() {
    if (!g->full) {
        g->style = GetWindowLongW(g->hwnd, GWL_STYLE);
        g->placement.length = sizeof g->placement;
        GetWindowPlacement(g->hwnd, &g->placement);
        MONITORINFO mi = {}; mi.cbSize = sizeof mi;
        GetMonitorInfoW(MonitorFromWindow(g->hwnd, MONITOR_DEFAULTTONEAREST), &mi);
        SetMenu(g->hwnd, nullptr);
        SetWindowLongW(g->hwnd, GWL_STYLE, g->style & ~WS_OVERLAPPEDWINDOW);
        SetWindowPos(g->hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        g->full = true;
    } else {
        SetWindowLongW(g->hwnd, GWL_STYLE, g->style);
        SetWindowPlacement(g->hwnd, &g->placement);
        SetWindowPos(g->hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        SetMenu(g->hwnd, g->bar);
        g->full = false;
    }
}

inline void enforce_min_size() {
    if (g->full || IsZoomed(g->hwnd) || IsIconic(g->hwnd)) return;
    RECT wr;
    GetWindowRect(g->hwnd, &wr);
    SetWindowPos(g->hwnd, nullptr, 0, 0, wr.right - wr.left, wr.bottom - wr.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
}

inline void zoom_by(float factor) {
    g->app->set_ui_zoom(g->app->ui_zoom() * factor);
    enforce_min_size();
    redraw();
}

inline void run_requests();

inline void run_command(int id) {
    switch (id) {
        case CMD_OPEN: {
            std::string p = pick_open(g->hwnd, g->cfg);
            if (!p.empty()) open_path(p);
            break;
        }
        case CMD_SAVE_PNG: {
            std::string p = pick_save(g->hwnd, g->cfg, g->cfg.app_name + ".png");
            if (!p.empty()) screenshot(p);
            redraw();
            break;
        }
        case CMD_EXIT: DestroyWindow(g->hwnd); break;
        case CMD_ZOOM_IN: zoom_by(1.1f); break;
        case CMD_ZOOM_OUT: zoom_by(1.0f / 1.1f); break;
        case CMD_ZOOM_RESET: g->app->set_ui_zoom(1.0f); enforce_min_size(); redraw(); break;
        case CMD_FULLSCREEN: toggle_fullscreen(); redraw(); break;
        default:
            g->app->command(id);
            refresh_chrome();
            redraw();
            run_requests();
            break;
    }
}

inline void run_requests() {
    for (int id = g->app->take_request(); id != 0; id = g->app->take_request()) run_command(id);
}

inline LRESULT CALLBACK proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (!g || !g->app) return DefWindowProcW(hwnd, msg, wp, lp);
    ShellApp& a = *g->app;
    if (g->cfg.window_test) {
        switch (msg) {
            case WM_DROPFILES: g->trace += 'D'; break;
            case WM_MOUSEMOVE: g->trace += 'm'; break;
            case WM_LBUTTONDOWN: g->trace += 'p'; break;
            case WM_LBUTTONUP: g->trace += 'r'; break;
            case WM_KEYDOWN: g->trace += 'k'; break;
            case WM_COMMAND: g->trace += 'c'; break;
            default: break;
        }
    }
    try {
        switch (msg) {
            case WM_ERASEBKGND: return 1;
            case WM_GETMINMAXINFO: {
                if (g->cfg.min_width <= 0 || g->cfg.min_height <= 0) break;
                const float zoom = a.ui_zoom();
                RECT rc = {0, 0, std::min(static_cast<int>(g->cfg.min_width * zoom), GetSystemMetrics(SM_CXSCREEN) * 95 / 100),
                           std::min(static_cast<int>(g->cfg.min_height * zoom), GetSystemMetrics(SM_CYSCREEN) * 90 / 100)};
                AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, TRUE);
                MINMAXINFO* mm = reinterpret_cast<MINMAXINFO*>(lp);
                mm->ptMinTrackSize.x = rc.right - rc.left;
                mm->ptMinTrackSize.y = rc.bottom - rc.top;
                return 0;
            }
            case WM_PAINT: {
                PAINTSTRUCT ps;
                BeginPaint(hwnd, &ps);
                paint();
                EndPaint(hwnd, &ps);
                return 0;
            }
            case WM_SIZE:
                if (wp != SIZE_MINIMIZED && LOWORD(lp) > 64 && HIWORD(lp) > 64) {
                    a.canvas().resize(LOWORD(lp), HIWORD(lp));
                    redraw();
                }
                return 0;
            case WM_MOUSEMOVE: a.move(static_cast<short>(LOWORD(lp)), static_cast<short>(HIWORD(lp))); redraw(); return 0;
            case WM_LBUTTONDOWN:
                ensure_fresh(); SetCapture(hwnd);
                a.press(static_cast<short>(LOWORD(lp)), static_cast<short>(HIWORD(lp)));
                redraw(); return 0;
            case WM_LBUTTONUP:
                ensure_fresh(); ReleaseCapture();
                a.release(static_cast<short>(LOWORD(lp)), static_cast<short>(HIWORD(lp)));
                refresh_chrome(); redraw(); run_requests(); return 0;
            case WM_MOUSEWHEEL: {
                POINT pt = {static_cast<short>(LOWORD(lp)), static_cast<short>(HIWORD(lp))};
                ScreenToClient(hwnd, &pt);
                a.move(pt.x, pt.y);
                if (GET_KEYSTATE_WPARAM(wp) & MK_CONTROL) zoom_by(GET_WHEEL_DELTA_WPARAM(wp) > 0 ? 1.1f : 1.0f / 1.1f);
                else a.wheel(pt.x, pt.y, GET_WHEEL_DELTA_WPARAM(wp), false);
                refresh_chrome(); redraw(); return 0;
            }
            case WM_KEYDOWN: {
                int vk = static_cast<int>(wp);
                bool shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0, ctrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
                if (ctrl && vk == 'O') { run_command(CMD_OPEN); return 0; }
                if (ctrl && vk == 'S') { run_command(CMD_SAVE_PNG); return 0; }
                if (ctrl && (vk == VK_OEM_PLUS || vk == VK_ADD)) { zoom_by(1.1f); return 0; }
                if (ctrl && (vk == VK_OEM_MINUS || vk == VK_SUBTRACT)) { zoom_by(1.0f / 1.1f); return 0; }
                if (ctrl && (vk == '0' || vk == VK_NUMPAD0)) { run_command(CMD_ZOOM_RESET); return 0; }
                if (vk == VK_F11) { toggle_fullscreen(); redraw(); return 0; }
                if (vk == 'Q' && !ctrl) { DestroyWindow(hwnd); return 0; }
                if (vk == VK_ESCAPE) {
                    if (a.escape()) { refresh_chrome(); redraw(); }
                    else if (g->full) { toggle_fullscreen(); redraw(); }
                    else if (g->cfg.quit_on_escape) DestroyWindow(hwnd);
                    return 0;
                }
                if (vk == 'S' && !ctrl) { screenshot(""); redraw(); return 0; }
                a.key(vk, ctrl, shift);
                refresh_chrome(); redraw(); run_requests(); return 0;
            }
            case WM_DROPFILES: {
                HDROP drop = reinterpret_cast<HDROP>(wp);
                wchar_t path[MAX_PATH * 4];
                if (DragQueryFileW(drop, 0, path, MAX_PATH * 4) > 0) open_path(narrow(path));
                DragFinish(drop);
                return 0;
            }
            case WM_COMMAND:
                if (HIWORD(wp) == 0) run_command(static_cast<int>(LOWORD(wp)));
                return 0;
            case WM_TIMER: {
                if (wp == 2) {
                    RECT cr;
                    GetClientRect(hwnd, &cr);
                    std::printf("window test: %d frames drawn, client %ldx%ld, foreground %d, trace %s, drawing %.1f ms and showing %.1f ms a frame, state %s\n", g->frames,
                                cr.right, cr.bottom, GetForegroundWindow() == hwnd ? 1 : 0, g->trace.c_str(), g->frames ? g->render_ms / g->frames : 0.0,
                                g->shown ? g->blit_ms / g->shown : 0.0, a.state_json().c_str());
                    DestroyWindow(hwnd);
                    return 0;
                }
                return 0;
            }
            case WM_DESTROY: PostQuitMessage(0); return 0;
            default: break;
        }
    } catch (const std::exception& e) {
        g->failed = true;
        MessageBoxA(hwnd, e.what(), "error", MB_OK | MB_ICONERROR);
        DestroyWindow(hwnd);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace shell_detail

inline int run_shell(ShellApp& app, const ShellConfig& cfg) {
    shell_detail::State st;
    st.app = &app;
    st.cfg = cfg;
    shell_detail::g = &st;
    WNDCLASSW wc = {};
    wc.lpfnWndProc = shell_detail::proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wc.lpszClassName = cfg.window_class.c_str();
    if (!RegisterClassW(&wc)) throw std::runtime_error("cannot register the window class");
    RECT rc = {0, 0, cfg.width, cfg.height};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, TRUE);
    RECT work = {0, 0, 1920, 1080};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    const int outer_w = std::min<int>(rc.right - rc.left, (work.right - work.left) * 96 / 100);
    const int outer_h = std::min<int>(rc.bottom - rc.top, (work.bottom - work.top) * 96 / 100);
    const DWORD ex_style = WS_EX_ACCEPTFILES | (cfg.window_test ? WS_EX_NOACTIVATE : 0);
    const DWORD style = WS_OVERLAPPEDWINDOW | (cfg.window_test ? 0 : WS_VISIBLE);
    const int at = cfg.window_test ? -20000 : CW_USEDEFAULT;
    st.hwnd = CreateWindowExW(ex_style, wc.lpszClassName, widen(app.title()).c_str(), style, at, at, outer_w, outer_h, nullptr, nullptr, wc.hInstance, nullptr);
    if (!st.hwnd) throw std::runtime_error("cannot create the window");
    DragAcceptFiles(st.hwnd, TRUE);
    shell_detail::refresh_chrome();
    if (!cfg.first_file.empty()) shell_detail::open_path(cfg.first_file);
    ShowWindow(st.hwnd, cfg.window_test ? SW_SHOWNOACTIVATE : SW_SHOW);
    if (cfg.window_test) {
        if (!cfg.test_keep_size) SetWindowPos(st.hwnd, nullptr, 0, 0, 300, 200, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        RECT cr;
        GetClientRect(st.hwnd, &cr);
        int w = cr.right, h = cr.bottom;
        for (int i = 0; i < cfg.test_flood; ++i)
            PostMessageW(st.hwnd, WM_MOUSEMOVE, 0, MAKELPARAM(10 + i % std::max(1, w - 20), 10 + (i * 7) % std::max(1, h - 20)));
        if (!cfg.test_drop_file.empty()) {
            struct DropFilesBlock { DWORD pFiles; POINT pt; BOOL fNC; BOOL fWide; };
            const std::wstring name = widen(cfg.test_drop_file);
            const size_t bytes = sizeof(DropFilesBlock) + (name.size() + 2) * sizeof(wchar_t);
            HGLOBAL block = GlobalAlloc(GHND, bytes);
            if (!block) throw std::runtime_error("cannot allocate the dropped file's block");
            DropFilesBlock* drop = static_cast<DropFilesBlock*>(GlobalLock(block));
            drop->pFiles = sizeof(DropFilesBlock);
            drop->fWide = TRUE;
            std::memcpy(reinterpret_cast<char*>(drop) + sizeof(DropFilesBlock), name.c_str(), name.size() * sizeof(wchar_t));
            GlobalUnlock(block);
            PostMessageW(st.hwnd, WM_DROPFILES, reinterpret_cast<WPARAM>(block), 0);
        }
        PostMessageW(st.hwnd, WM_MOUSEMOVE, 0, MAKELPARAM(w / 2, h / 2));
        PostMessageW(st.hwnd, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(w / 2, h / 2));
        PostMessageW(st.hwnd, WM_MOUSEMOVE, MK_LBUTTON, MAKELPARAM(w / 2 + 50, h / 2 + 20));
        PostMessageW(st.hwnd, WM_LBUTTONUP, 0, MAKELPARAM(w / 2 + 50, h / 2 + 20));
        PostMessageW(st.hwnd, WM_KEYDOWN, VK_RIGHT, 0);
        PostMessageW(st.hwnd, WM_COMMAND, CMD_ZOOM_IN, 0);
        PostMessageW(st.hwnd, WM_KEYDOWN, VK_SPACE, 0);
        PostMessageW(st.hwnd, WM_KEYDOWN, 'S', 0);
        SetTimer(st.hwnd, 2, 1500, nullptr);
    }
    timeBeginPeriod(1);
    const double min_gap = 1.0 / 60.0;
    double last_frame_s = 0;
    st.last_tick_s = shell_detail::now_s();
    MSG msg;
    for (bool running = true; running;) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) { running = false; break; }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running) break;
        try {
            double now = shell_detail::now_s();
            const bool moving = app.tick(std::min(now - st.last_tick_s, 0.2));
            st.last_tick_s = now;
            if (moving) st.dirty = true;
            if (st.dirty && now - last_frame_s >= min_gap) {
                shell_detail::render_now();
                shell_detail::blit_now();
                last_frame_s = shell_detail::now_s();
            } else if (st.shown_stale) {
                shell_detail::blit_now();
            }
            now = shell_detail::now_s();
            const double until_next = last_frame_s + min_gap - now;
            DWORD wait_ms = 100;
            if (st.dirty || moving) wait_ms = until_next > 0 ? static_cast<DWORD>(std::ceil(until_next * 1000.0)) : 0;
            MsgWaitForMultipleObjectsEx(0, nullptr, wait_ms, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
        } catch (const std::exception& e) {
            st.failed = true;
            MessageBoxA(st.hwnd, e.what(), "error", MB_OK | MB_ICONERROR);
            DestroyWindow(st.hwnd);
        }
    }
    timeEndPeriod(1);
    shell_detail::g = nullptr;
    return st.failed ? 1 : 0;
}

}  // namespace vitro
