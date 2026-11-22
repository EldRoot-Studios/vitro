// vitro.exe: the Vitro app in one window.
//
// The interface is the web app of web/ (React), shown by the Microsoft Edge engine that Windows already has
// (WebView2). The program carries the app, the sample traces and the licence notices inside itself and answers the
// app's requests itself, so there is no server, no port and no browser: the window asks https://vitro.example/ for
// a file and this program hands it over. Traces are opened from the command line, by dropping them on the window or
// on the program's icon, or with the picker in the app; the app reads them in the window, nothing is uploaded.
//
//   vitro.exe [file.vtdump] [file.vtdemo] ...
//   vitro.exe --help
//
// Build: bash build.sh app   (needs the WebView2 SDK, which tools/fetch_webview2.py downloads once)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dwmapi.h>
#include <objbase.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shlwapi.h>

#include "WebView2.h"
#include "webview2_uuids.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "app_assets.hpp"
#include "app_library.hpp"

#ifndef VITRO_VERSION
#define VITRO_VERSION "0.0.0"
#endif

namespace {

using namespace vitro::app;

// The address the window believes it is talking to. ".example" can never resolve, so even a request that slipped past
// the interception below could not reach another machine.
constexpr const wchar_t* kOrigin = L"https://vitro.example";

// What the app may do: only itself. The same policy `vitro web` sends.
constexpr const wchar_t* kPolicy =
    L"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
    L"connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

constexpr int kExitUsage = 2;
constexpr int kExitFailed = 3;

// ---- small COM helpers: this program does not use WRL, which MinGW does not have ----

template <class T>
class ComPtr {
public:
    ComPtr() = default;
    ComPtr(const ComPtr& o) : p_(o.p_) { if (p_) p_->AddRef(); }
    ComPtr(ComPtr&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    ComPtr& operator=(ComPtr o) noexcept { std::swap(p_, o.p_); return *this; }
    ~ComPtr() { reset(); }
    void reset() { if (p_) { p_->Release(); p_ = nullptr; } }
    T* get() const { return p_; }
    T* operator->() const { return p_; }
    explicit operator bool() const { return p_ != nullptr; }
    // For an out-parameter: releases what is held and gives the address to write the new pointer to.
    T** put() { reset(); return &p_; }
    // Takes a reference to `raw` (which COM handed out for the duration of a callback only).
    void share(T* raw) { if (raw) raw->AddRef(); T* old = p_; p_ = raw; if (old) old->Release(); }
    template <class U>
    ComPtr<U> as() const {
        ComPtr<U> out;
        if (p_) p_->QueryInterface(__uuidof(U), reinterpret_cast<void**>(out.put()));
        return out;
    }

private:
    T* p_ = nullptr;
};

// A COM callback object that calls a function. `Interface` is a WebView2 *Handler interface whose only method is
// Invoke(Args...).
template <class Interface, class... Args>
class Callback final : public Interface {
public:
    using Fn = std::function<HRESULT(Args...)>;
    explicit Callback(Fn fn) : fn_(std::move(fn)) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown) || IsEqualIID(riid, __uuidof(Interface))) {
            *out = static_cast<Interface*>(this);
            AddRef();
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&refs_)); }
    ULONG STDMETHODCALLTYPE Release() override {
        const LONG n = InterlockedDecrement(&refs_);
        if (n == 0) delete this;
        return static_cast<ULONG>(n);
    }
    HRESULT STDMETHODCALLTYPE Invoke(Args... args) override { return fn_(args...); }

private:
    Fn fn_;
    LONG refs_ = 1;
};

// Makes a callback, hands it to `use` (which registers it and so takes its own reference), and drops ours.
template <class Interface, class... Args, class F, class U>
HRESULT with_callback(F&& fn, U&& use) {
    auto* cb = new Callback<Interface, Args...>(std::forward<F>(fn));
    const HRESULT hr = use(cb);
    cb->Release();
    return hr;
}

// ---- talking to the person: console when there is one, a message box otherwise ----

HANDLE console_handle(DWORD which) {
    HANDLE h = GetStdHandle(which);
    if (h && h != INVALID_HANDLE_VALUE) return h;
    static bool attached = AttachConsole(ATTACH_PARENT_PROCESS) != 0;
    if (!attached) return nullptr;
    h = CreateFileW(L"CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    return h == INVALID_HANDLE_VALUE ? nullptr : h;
}

bool write_text(DWORD which, const std::string& utf8_text) {
    HANDLE h = console_handle(which);
    if (!h) return false;
    DWORD mode = 0;
    DWORD n = 0;
    if (GetConsoleMode(h, &mode)) {
        const std::wstring w = wide(utf8_text);
        return WriteConsoleW(h, w.data(), static_cast<DWORD>(w.size()), &n, nullptr) != 0;
    }
    return WriteFile(h, utf8_text.data(), static_cast<DWORD>(utf8_text.size()), &n, nullptr) != 0;
}

void say(const std::string& text) { write_text(STD_OUTPUT_HANDLE, text); }

// An error: on stderr when the program was started from a console or with its output redirected, else in a box, so
// that a double-click never fails silently.
void complain(const std::string& text) {
    if (!write_text(STD_ERROR_HANDLE, "vitro: " + text + "\n")) MessageBoxW(nullptr, wide(text).c_str(), L"Vitro", MB_OK | MB_ICONERROR);
}

// ---- the command line ----

struct Options {
    std::vector<std::wstring> files;
    bool devtools = false;
    bool help = false;
    bool version = false;
    bool licenses = false;
    std::wstring eval_file;   // run this script in the page and print what it returns (for automation)
    std::wstring screenshot;  // save a picture of the window to this PNG file
    std::wstring hash;        // open the app at this #fragment
    int width = 1440, height = 900;  // size of the window's client area, in device-independent pixels
    int wait_ms = 500;        // settle time before a screenshot
    int timeout_ms = 60000;   // longest an automated run may take
};

const char* kHelp =
    "Vitro: a glass-box inspector for PyTorch traces.\n"
    "\n"
    "usage: vitro.exe [options] [file.vtdump | file.vtdemo ...]\n"
    "\n"
    "Opens the app. A dump (.vtdump) powers the inspector tabs and a recording (.vtdemo) the replay; give both files\n"
    "of one run to use every tab. Files can also be dropped on the window or on this program, or chosen in the app.\n"
    "The sample traces are built in. Nothing is uploaded: the app reads the files in its window.\n"
    "\n"
    "options:\n"
    "  --help, -h        show this text\n"
    "  --version         show the version\n"
    "  --licenses        show the licences of what is built in\n"
    "  --devtools        enable the browser developer tools (F12)\n"
    "\n"
    "for automation and tests:\n"
    "  --size WxH        window client size in pixels at 100% scale (default 1440x900)\n"
    "  --hash FRAGMENT   open the app at this address fragment, e.g. \"#open=0,1&view=net\"\n"
    "  --eval-file FILE  run the JavaScript in FILE in the page, print its result as JSON, exit\n"
    "  --screenshot PNG  save a picture of the window and exit (after --eval-file, if both are given)\n"
    "  --wait MS         wait this long before the picture (default 500)\n"
    "  --timeout MS      give up after this long (default 60000)\n"
    "\n"
    "The settings and the engine's cache live in %LOCALAPPDATA%\\Vitro, or in the folder named by VITRO_DATA_DIR.\n"
    "The program needs the Microsoft Edge WebView2 Runtime, which comes with Windows 11 and recent Windows 10.\n";

// Parses the command line; throws std::runtime_error with the message for the person.
Options parse_command_line() {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) throw std::runtime_error("cannot read the command line");
    std::unique_ptr<LPWSTR, decltype(&LocalFree)> guard(argv, &LocalFree);
    Options o;
    auto value = [&](int& i, const std::wstring& flag) -> std::wstring {
        if (i + 1 >= argc) throw std::runtime_error(utf8(flag) + " needs a value (see --help)");
        return argv[++i];
    };
    auto number = [&](const std::wstring& text, const std::wstring& flag) -> int {
        try {
            size_t used = 0;
            const int n = std::stoi(text, &used);
            if (used != text.size() || n <= 0) throw std::invalid_argument("x");
            return n;
        } catch (const std::logic_error&) {
            throw std::runtime_error(utf8(flag) + " needs a positive whole number, not \"" + utf8(text) + "\"");
        }
    };
    for (int i = 1; i < argc; ++i) {
        const std::wstring a = argv[i];
        if (a == L"--help" || a == L"-h" || a == L"/?") o.help = true;
        else if (a == L"--version") o.version = true;
        else if (a == L"--licenses") o.licenses = true;
        else if (a == L"--devtools") o.devtools = true;
        else if (a == L"--eval-file") o.eval_file = value(i, a);
        else if (a == L"--screenshot") o.screenshot = value(i, a);
        else if (a == L"--hash") o.hash = value(i, a);
        else if (a == L"--wait") o.wait_ms = number(value(i, a), a);
        else if (a == L"--timeout") o.timeout_ms = number(value(i, a), a);
        else if (a == L"--size") {
            const std::wstring v = value(i, a);
            const size_t x = v.find_first_of(L"xX");
            if (x == std::wstring::npos) throw std::runtime_error("--size needs WIDTHxHEIGHT, e.g. 1440x900");
            o.width = number(v.substr(0, x), a);
            o.height = number(v.substr(x + 1), a);
        } else if (a.size() > 1 && a[0] == L'-') throw std::runtime_error("unknown option " + utf8(a) + " (see --help)");
        else o.files.push_back(a);
    }
    return o;
}

// ---- reading what is built in ----

std::string read_file_text(const std::wstring& path) {
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + utf8(path) + " (Windows error " + std::to_string(GetLastError()) + ")");
    std::string text;
    char buf[65536];
    DWORD n = 0;
    while (ReadFile(f, buf, sizeof buf, &n, nullptr) && n > 0) text.append(buf, n);
    CloseHandle(f);
    return text;
}

std::wstring env_var(const wchar_t* name) {
    wchar_t buf[2048];
    const DWORD n = GetEnvironmentVariableW(name, buf, 2048);
    return n > 0 && n < 2048 ? std::wstring(buf, n) : std::wstring();
}

// Where the program keeps what it needs between runs: the engine's profile and the unpacked loader.
std::wstring data_dir() {
    std::wstring dir = env_var(L"VITRO_DATA_DIR");
    if (dir.empty()) {
        const std::wstring local = env_var(L"LOCALAPPDATA");
        if (local.empty()) throw std::runtime_error("neither VITRO_DATA_DIR nor LOCALAPPDATA is set, so there is nowhere to keep the engine's data");
        dir = local + L"\\Vitro";
    }
    const int rc = SHCreateDirectoryExW(nullptr, dir.c_str(), nullptr);
    if (rc != ERROR_SUCCESS && rc != ERROR_ALREADY_EXISTS && rc != ERROR_FILE_EXISTS)
        throw std::runtime_error("cannot create " + utf8(dir) + " (Windows error " + std::to_string(rc) + ")");
    return dir;
}

std::wstring exe_dir() {
    wchar_t buf[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(sizeof buf / sizeof buf[0]));
    std::wstring p(buf, n);
    return p.substr(0, p.rfind(L'\\'));
}

bool file_has_size(const std::wstring& path, uint64_t size) {
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &a)) return false;
    return ((static_cast<uint64_t>(a.nFileSizeHigh) << 32) | a.nFileSizeLow) == size;
}

// WebView2Loader.dll is what finds the installed engine. It is built into this program; it is unpacked once and
// loaded from there. A copy placed next to the program is used instead, if there is one.
HMODULE load_loader(const std::vector<Item>& items) {
    const std::wstring beside = exe_dir() + L"\\WebView2Loader.dll";
    if (GetFileAttributesW(beside.c_str()) != INVALID_FILE_ATTRIBUTES) {
        if (HMODULE m = LoadLibraryW(beside.c_str())) return m;
        throw std::runtime_error("cannot load " + utf8(beside) + " (Windows error " + std::to_string(GetLastError()) + ")");
    }
    const Item* loader = nullptr;
    for (const Item& it : items) if (it.kind == "loader") loader = &it;
    if (!loader) throw std::runtime_error("this vitro.exe does not contain WebView2Loader.dll, and none lies next to it");
    const Blob blob = resource(loader->id);
    if (!blob.data) throw std::runtime_error("the built-in WebView2Loader.dll is missing from the program");
    const std::wstring dir = data_dir() + L"\\runtime";
    SHCreateDirectoryExW(nullptr, dir.c_str(), nullptr);
    const std::wstring target = dir + L"\\WebView2Loader-" + wide(loader->name) + L".dll";
    if (!file_has_size(target, blob.size)) {
        const std::wstring part = target + L"." + std::to_wstring(GetCurrentProcessId()) + L".part";
        HANDLE f = CreateFileW(part.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot write " + utf8(part) + " (Windows error " + std::to_string(GetLastError()) + ")");
        DWORD n = 0;
        const bool ok = WriteFile(f, blob.data, static_cast<DWORD>(blob.size), &n, nullptr) && n == blob.size;
        CloseHandle(f);
        if (!ok) {
            DeleteFileW(part.c_str());
            throw std::runtime_error("cannot write " + utf8(part));
        }
        // Another copy of the program may be running and have the file loaded; then it is already the right one.
        if (!MoveFileExW(part.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING) && !file_has_size(target, blob.size)) {
            DeleteFileW(part.c_str());
            throw std::runtime_error("cannot place " + utf8(target) + " (Windows error " + std::to_string(GetLastError()) + ")");
        }
        DeleteFileW(part.c_str());
    }
    if (HMODULE m = LoadLibraryW(target.c_str())) return m;
    throw std::runtime_error("cannot load " + utf8(target) + " (Windows error " + std::to_string(GetLastError()) + ")");
}

// ---- answering the window's requests ----

std::string percent_decode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        const auto hex = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        if (s[i] == '%' && i + 2 < s.size() + 0 && hex(s[i + 1]) >= 0 && hex(s[i + 2]) >= 0) {
            out += static_cast<char>(hex(s[i + 1]) * 16 + hex(s[i + 2]));
            i += 2;
        } else {
            out += s[i];
        }
    }
    return out;
}

struct Reply {
    int status = 404;
    std::string mime = "text/plain; charset=utf-8";
    ComPtr<IStream> body;
    std::string allow;  // for a 405
};

const char* reason_for(int status) {
    switch (status) {
        case 200: return "OK";
        case 405: return "Method Not Allowed";
        default: return "Not Found";
    }
}

ComPtr<IStream> text_stream(const std::string& text) {
    ComPtr<IStream> s;
    s.share(SHCreateMemStream(reinterpret_cast<const BYTE*>(text.data()), static_cast<UINT>(text.size())));
    return s;
}

class Router {
public:
    Router(std::vector<Entry> entries, std::vector<Item> items) : entries_(std::move(entries)), items_(std::move(items)) {
        for (const Item& it : items_) if (it.kind == "asset") assets_[it.name] = &it;
    }
    const std::vector<Entry>& entries() const { return entries_; }

    // Answers a request for `path` (already cut at ? and #, still percent-encoded). Same rules as `vitro web`.
    Reply answer(const std::string& method, const std::string& raw_path) const {
        Reply r;
        const bool head = method == "HEAD";
        if (method != "GET" && !head) {
            r.status = 405;
            r.allow = "GET, HEAD";
            r.body = text_stream("405 only GET and HEAD are served\n");
            return r;
        }
        const std::string path = percent_decode(raw_path);
        if (path == "/api/library") return ok("application/json; charset=utf-8", library_json(entries_), head);
        if (path.compare(0, 7, "/files/") == 0) return file(path.substr(7), head);
        return asset(path, head);
    }

private:
    Reply ok(const std::string& mime, const std::string& body, bool head) const {
        Reply r;
        r.status = 200;
        r.mime = mime;
        r.body = text_stream(head ? std::string() : body);
        return r;
    }
    Reply missing(const char* what) const {
        Reply r;
        r.body = text_stream(std::string("404 ") + what + "\n");
        return r;
    }

    Reply file(const std::string& tail, bool head) const {
        if (tail.empty() || tail.size() > 9) return missing("no such trace file");
        for (char c : tail) if (c < '0' || c > '9') return missing("no such trace file");
        const size_t id = static_cast<size_t>(std::stoul(tail));
        if (id >= entries_.size()) return missing("no such trace file");
        const Entry& e = entries_[id];
        Reply r;
        r.mime = "application/octet-stream";
        if (e.resource_id != 0) {
            const Blob b = resource(e.resource_id);
            r.body.share(SHCreateMemStream(b.data, head ? 0 : static_cast<UINT>(b.size)));
        } else if (head) {
            r.body = text_stream("");
        } else if (FAILED(SHCreateStreamOnFileEx(e.path.c_str(), STGM_READ | STGM_SHARE_DENY_WRITE, FILE_ATTRIBUTE_NORMAL, FALSE, nullptr, r.body.put()))) {
            return missing("cannot read the trace file");
        }
        r.status = 200;
        return r;
    }

    Reply asset(const std::string& path, bool head) const {
        const std::string rel = path == "/" ? "index.html" : (path.empty() ? std::string() : path.substr(1));
        // each segment must be a plain name: no empty, ".", "..", backslash, colon or NUL
        size_t from = 0;
        while (true) {
            const size_t slash = rel.find('/', from);
            const std::string part = rel.substr(from, slash == std::string::npos ? std::string::npos : slash - from);
            if (part.empty() || part == "." || part == ".." || part.find_first_of("\\:") != std::string::npos || part.find('\0') != std::string::npos)
                return missing("not found");
            if (slash == std::string::npos) break;
            from = slash + 1;
        }
        const auto it = assets_.find(rel);
        if (it == assets_.end()) return missing("not found");
        const Blob b = resource(it->second->id);
        if (!b.data) return missing("not found");
        Reply r;
        r.status = 200;
        r.mime = it->second->mime;
        r.body.share(SHCreateMemStream(b.data, head ? 0 : static_cast<UINT>(b.size)));
        return r;
    }

    std::vector<Entry> entries_;
    std::vector<Item> items_;
    std::unordered_map<std::string, const Item*> assets_;
};

// ---- the window ----

class App {
public:
    App(Options options, Router router, std::vector<Item> items) : opt_(std::move(options)), router_(std::move(router)), items_(std::move(items)) {}

    int run(HINSTANCE instance) {
        create_window(instance);
        const HRESULT hr = start_engine();
        if (FAILED(hr)) fail(engine_message(hr), kExitFailed);
        MSG msg;
        while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        return exit_code_;
    }

private:
    static LRESULT CALLBACK proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        App* self = reinterpret_cast<App*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (msg == WM_NCCREATE) {
            self = static_cast<App*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
            self->hwnd_ = hwnd;
        }
        return self ? self->message(hwnd, msg, wp, lp) : DefWindowProcW(hwnd, msg, wp, lp);
    }

    LRESULT message(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        switch (msg) {
            case WM_SIZE:
                resize();
                return 0;
            case WM_MOVE:
            case WM_MOVING:
                if (controller_) controller_->NotifyParentWindowPositionChanged();
                return 0;
            case WM_SETFOCUS:
                if (controller_) controller_->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
                return 0;
            case WM_ERASEBKGND:
                return 1;  // the engine paints everything; erasing would flash
            case WM_GETMINMAXINFO: {
                const UINT dpi = GetDpiForWindow(hwnd);
                auto* mm = reinterpret_cast<MINMAXINFO*>(lp);
                mm->ptMinTrackSize.x = MulDiv(480, static_cast<int>(dpi), 96);
                mm->ptMinTrackSize.y = MulDiv(360, static_cast<int>(dpi), 96);
                return 0;
            }
            case WM_DPICHANGED: {
                const RECT* r = reinterpret_cast<const RECT*>(lp);
                SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
                return 0;
            }
            case WM_TIMER:
                if (auto it = timers_.find(wp); it != timers_.end()) {
                    KillTimer(hwnd, wp);
                    auto fn = std::move(it->second);
                    timers_.erase(it);
                    fn();
                }
                return 0;
            case WM_CLOSE:
                DestroyWindow(hwnd);
                return 0;
            case WM_DESTROY:
                controller_.reset();
                PostQuitMessage(exit_code_);
                return 0;
            default:
                return DefWindowProcW(hwnd, msg, wp, lp);
        }
    }

    void create_window(HINSTANCE instance) {
        WNDCLASSEXW wc = {};
        wc.cbSize = sizeof wc;
        wc.lpfnWndProc = &App::proc;
        wc.hInstance = instance;
        wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(1));
        wc.hIconSm = wc.hIcon;
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hbrBackground = CreateSolidBrush(RGB(38, 38, 36));  // the app's grey, so the first moments are not white
        wc.lpszClassName = L"VitroWindow";
        if (!RegisterClassExW(&wc)) throw std::runtime_error("cannot register the window class (Windows error " + std::to_string(GetLastError()) + ")");
        const DWORD style = WS_OVERLAPPEDWINDOW;
        const UINT dpi = GetDpiForSystem();
        RECT r = {0, 0, MulDiv(opt_.width, static_cast<int>(dpi), 96), MulDiv(opt_.height, static_cast<int>(dpi), 96)};
        AdjustWindowRectExForDpi(&r, style, FALSE, 0, dpi);
        RECT work;
        SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
        const int w = r.right - r.left;
        const int h = r.bottom - r.top;
        const int x = work.left + std::max<LONG>(0, ((work.right - work.left) - w) / 2);
        const int y = work.top + std::max<LONG>(0, ((work.bottom - work.top) - h) / 2);
        hwnd_ = CreateWindowExW(0, L"VitroWindow", L"Vitro", style, x, y, w, h, nullptr, nullptr, instance, this);
        if (!hwnd_) throw std::runtime_error("cannot create the window (Windows error " + std::to_string(GetLastError()) + ")");
        // A dark title bar to match the app. These need Windows 10 1809 or 11 and are only looks: on an older Windows
        // the call fails and the bar stays light, which is fine.
        const BOOL dark = TRUE;
        DwmSetWindowAttribute(hwnd_, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof dark);
        const COLORREF caption = RGB(38, 38, 36);
        const COLORREF text = RGB(250, 249, 245);
        DwmSetWindowAttribute(hwnd_, 35 /* DWMWA_CAPTION_COLOR */, &caption, sizeof caption);
        DwmSetWindowAttribute(hwnd_, 36 /* DWMWA_TEXT_COLOR */, &text, sizeof text);
        ShowWindow(hwnd_, SW_SHOWNORMAL);
        UpdateWindow(hwnd_);
    }

    void resize() {
        if (!controller_) return;
        RECT r;
        GetClientRect(hwnd_, &r);
        controller_->put_Bounds(r);
    }

    void after(UINT ms, std::function<void()> fn) {
        const UINT_PTR id = next_timer_++;
        timers_[id] = std::move(fn);
        SetTimer(hwnd_, id, ms, nullptr);
    }

    void fail(const std::string& message, int code) {
        complain(message);
        exit_code_ = code;
        if (hwnd_) DestroyWindow(hwnd_);
        else PostQuitMessage(code);
    }

    void finish(int code) {
        exit_code_ = code;
        DestroyWindow(hwnd_);
    }

    static std::string engine_message(HRESULT hr) {
        if (hr == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND) || hr == HRESULT_FROM_WIN32(ERROR_PRODUCT_UNINSTALLED))
            return "Vitro needs the Microsoft Edge WebView2 Runtime, and it is not installed. It comes with Windows 11 and "
                   "with recent updates of Windows 10; otherwise get it from https://developer.microsoft.com/microsoft-edge/webview2/";
        char code[32];
        std::snprintf(code, sizeof code, "0x%08lX", static_cast<unsigned long>(hr));
        return std::string("the Microsoft Edge WebView2 engine could not be started (error ") + code + ")";
    }

    HRESULT start_engine() {
        HMODULE loader = load_loader(items_);
        using Create = HRESULT(STDAPICALLTYPE*)(PCWSTR, PCWSTR, ICoreWebView2EnvironmentOptions*, ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler*);
        auto create = reinterpret_cast<Create>(reinterpret_cast<void*>(GetProcAddress(loader, "CreateCoreWebView2EnvironmentWithOptions")));
        if (!create) throw std::runtime_error("the WebView2 loader has no CreateCoreWebView2EnvironmentWithOptions");
        const std::wstring profile = data_dir() + L"\\WebView2";
        return with_callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler, HRESULT, ICoreWebView2Environment*>(
            [this](HRESULT hr, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(hr)) {
                    fail(engine_message(hr), kExitFailed);
                    return S_OK;
                }
                env_.share(env);
                const HRESULT made = with_callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler, HRESULT, ICoreWebView2Controller*>(
                    [this](HRESULT hr2, ICoreWebView2Controller* controller) -> HRESULT {
                        if (FAILED(hr2)) {
                            fail(engine_message(hr2), kExitFailed);
                            return S_OK;
                        }
                        controller_.share(controller);
                        return engine_ready();
                    },
                    [&](auto* cb) { return env_->CreateCoreWebView2Controller(hwnd_, cb); });
                if (FAILED(made)) fail(engine_message(made), kExitFailed);
                return S_OK;
            },
            [&](auto* cb) { return create(nullptr, profile.c_str(), nullptr, cb); });
    }

    // The engine exists: make it the app and open the page.
    HRESULT engine_ready() {
        if (FAILED(controller_->get_CoreWebView2(view_.put()))) {
            fail("the WebView2 engine gave no view", kExitFailed);
            return S_OK;
        }
        if (auto c2 = controller_.as<ICoreWebView2Controller2>()) {
            const COREWEBVIEW2_COLOR grey = {255, 38, 38, 36};
            c2->put_DefaultBackgroundColor(grey);
        }
        configure_settings();
        resize();
        controller_->put_IsVisible(TRUE);

        EventRegistrationToken token;
        // Everything the page asks of https://vitro.example/ is answered here.
        view_->AddWebResourceRequestedFilter((std::wstring(kOrigin) + L"/*").c_str(), COREWEBVIEW2_WEB_RESOURCE_CONTEXT_ALL);
        with_callback<ICoreWebView2WebResourceRequestedEventHandler, ICoreWebView2*, ICoreWebView2WebResourceRequestedEventArgs*>(
            [this](ICoreWebView2*, ICoreWebView2WebResourceRequestedEventArgs* args) -> HRESULT { return on_request(args); },
            [&](auto* cb) { return view_->add_WebResourceRequested(cb, &token); });
        // The page stays on its own address: nothing it links to opens inside this window.
        with_callback<ICoreWebView2NavigationStartingEventHandler, ICoreWebView2*, ICoreWebView2NavigationStartingEventArgs*>(
            [](ICoreWebView2*, ICoreWebView2NavigationStartingEventArgs* args) -> HRESULT {
                LPWSTR uri = nullptr;
                args->get_Uri(&uri);
                const std::wstring u = uri ? uri : L"";
                CoTaskMemFree(uri);
                if (u.compare(0, wcslen(kOrigin), kOrigin) != 0 && u != L"about:blank") args->put_Cancel(TRUE);
                return S_OK;
            },
            [&](auto* cb) { return view_->add_NavigationStarting(cb, &token); });
        with_callback<ICoreWebView2NewWindowRequestedEventHandler, ICoreWebView2*, ICoreWebView2NewWindowRequestedEventArgs*>(
            [](ICoreWebView2*, ICoreWebView2NewWindowRequestedEventArgs* args) -> HRESULT { return args->put_Handled(TRUE); },
            [&](auto* cb) { return view_->add_NewWindowRequested(cb, &token); });
        with_callback<ICoreWebView2PermissionRequestedEventHandler, ICoreWebView2*, ICoreWebView2PermissionRequestedEventArgs*>(
            [](ICoreWebView2*, ICoreWebView2PermissionRequestedEventArgs* args) -> HRESULT { return args->put_State(COREWEBVIEW2_PERMISSION_STATE_DENY); },
            [&](auto* cb) { return view_->add_PermissionRequested(cb, &token); });
        // The window's title is the page's title.
        with_callback<ICoreWebView2DocumentTitleChangedEventHandler, ICoreWebView2*, IUnknown*>(
            [this](ICoreWebView2* sender, IUnknown*) -> HRESULT {
                LPWSTR title = nullptr;
                if (SUCCEEDED(sender->get_DocumentTitle(&title)) && title) SetWindowTextW(hwnd_, title);
                CoTaskMemFree(title);
                return S_OK;
            },
            [&](auto* cb) { return view_->add_DocumentTitleChanged(cb, &token); });
        // A full-screen request of the page (the app's own button) takes the whole window out of its frame.
        with_callback<ICoreWebView2ContainsFullScreenElementChangedEventHandler, ICoreWebView2*, IUnknown*>(
            [this](ICoreWebView2* sender, IUnknown*) -> HRESULT {
                BOOL full = FALSE;
                sender->get_ContainsFullScreenElement(&full);
                set_fullscreen(full != FALSE);
                return S_OK;
            },
            [&](auto* cb) { return view_->add_ContainsFullScreenElementChanged(cb, &token); });
        with_callback<ICoreWebView2NavigationCompletedEventHandler, ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs*>(
            [this](ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs* args) -> HRESULT {
                BOOL ok = FALSE;
                args->get_IsSuccess(&ok);
                if (!ok) {
                    fail("the app could not be loaded into the window", kExitFailed);
                    return S_OK;
                }
                if (!loaded_) {
                    loaded_ = true;
                    if (!opt_.eval_file.empty() || !opt_.screenshot.empty()) automate();
                }
                return S_OK;
            },
            [&](auto* cb) { return view_->add_NavigationCompleted(cb, &token); });
        // F11 toggles full screen from the keyboard.
        with_callback<ICoreWebView2AcceleratorKeyPressedEventHandler, ICoreWebView2Controller*, ICoreWebView2AcceleratorKeyPressedEventArgs*>(
            [this](ICoreWebView2Controller*, ICoreWebView2AcceleratorKeyPressedEventArgs* args) -> HRESULT {
                COREWEBVIEW2_KEY_EVENT_KIND kind;
                UINT key = 0;
                args->get_KeyEventKind(&kind);
                args->get_VirtualKey(&key);
                if (key == VK_F11 && (kind == COREWEBVIEW2_KEY_EVENT_KIND_KEY_DOWN || kind == COREWEBVIEW2_KEY_EVENT_KIND_SYSTEM_KEY_DOWN)) {
                    set_fullscreen(!fullscreen_);
                    args->put_Handled(TRUE);
                }
                return S_OK;
            },
            [&](auto* cb) { return controller_->add_AcceleratorKeyPressed(cb, &token); });

        std::wstring url = std::wstring(kOrigin) + L"/";
        if (!opt_.hash.empty()) url += opt_.hash;
        else {
            std::string opened;
            for (size_t i = 0; i < router_.entries().size(); ++i)
                if (router_.entries()[i].source == "argument") opened += (opened.empty() ? "" : ",") + std::to_string(i);
            if (!opened.empty()) url += L"#open=" + wide(opened);
        }
        view_->Navigate(url.c_str());
        if (!opt_.eval_file.empty() || !opt_.screenshot.empty()) {
            after(static_cast<UINT>(opt_.timeout_ms), [this] { fail("timed out", kExitFailed); });
        }
        SetFocus(hwnd_);
        return S_OK;
    }

    void configure_settings() {
        ComPtr<ICoreWebView2Settings> s;
        if (FAILED(view_->get_Settings(s.put()))) return;
        s->put_AreDefaultContextMenusEnabled(opt_.devtools ? TRUE : FALSE);
        s->put_AreDevToolsEnabled(opt_.devtools ? TRUE : FALSE);
        s->put_IsStatusBarEnabled(FALSE);
        s->put_IsBuiltInErrorPageEnabled(FALSE);
        s->put_AreDefaultScriptDialogsEnabled(TRUE);
        s->put_IsWebMessageEnabled(FALSE);
        // These belong to newer versions of the engine's interfaces; an older runtime simply lacks them.
        if (auto s3 = s.as<ICoreWebView2Settings3>()) s3->put_AreBrowserAcceleratorKeysEnabled(opt_.devtools ? TRUE : FALSE);
        if (auto s4 = s.as<ICoreWebView2Settings4>()) {
            s4->put_IsPasswordAutosaveEnabled(FALSE);
            s4->put_IsGeneralAutofillEnabled(FALSE);
        }
        if (auto s6 = s.as<ICoreWebView2Settings6>()) s6->put_IsSwipeNavigationEnabled(FALSE);
    }

    HRESULT on_request(ICoreWebView2WebResourceRequestedEventArgs* args) {
        ComPtr<ICoreWebView2WebResourceRequest> req;
        if (FAILED(args->get_Request(req.put()))) return S_OK;
        LPWSTR uri = nullptr;
        LPWSTR method = nullptr;
        req->get_Uri(&uri);
        req->get_Method(&method);
        std::wstring u = uri ? uri : L"";
        const std::string verb = method ? utf8(method) : "GET";
        CoTaskMemFree(uri);
        CoTaskMemFree(method);
        if (u.compare(0, wcslen(kOrigin), kOrigin) != 0) return S_OK;
        std::wstring rest = u.substr(wcslen(kOrigin));
        const size_t cut = rest.find_first_of(L"?#");
        if (cut != std::wstring::npos) rest.resize(cut);
        const Reply reply = router_.answer(verb, utf8(rest));
        std::wstring headers = L"Content-Type: " + wide(reply.mime) + L"\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
                               L"Referrer-Policy: no-referrer\r\nContent-Security-Policy: " + std::wstring(kPolicy) + L"\r\n";
        if (!reply.allow.empty()) headers += L"Allow: " + wide(reply.allow) + L"\r\n";
        ComPtr<ICoreWebView2WebResourceResponse> response;
        if (SUCCEEDED(env_->CreateWebResourceResponse(reply.body.get(), reply.status, wide(reason_for(reply.status)).c_str(), headers.c_str(), response.put())))
            args->put_Response(response.get());
        return S_OK;
    }

    void set_fullscreen(bool on) {
        if (on == fullscreen_) return;
        fullscreen_ = on;
        if (on) {
            saved_.length = sizeof saved_;
            GetWindowPlacement(hwnd_, &saved_);
            MONITORINFO mi = {};
            mi.cbSize = sizeof mi;
            GetMonitorInfoW(MonitorFromWindow(hwnd_, MONITOR_DEFAULTTONEAREST), &mi);
            SetWindowLongPtrW(hwnd_, GWL_STYLE, WS_POPUP | WS_VISIBLE);
            SetWindowPos(hwnd_, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_FRAMECHANGED);
        } else {
            SetWindowLongPtrW(hwnd_, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
            SetWindowPlacement(hwnd_, &saved_);
            SetWindowPos(hwnd_, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        }
        // Asking the page to leave full screen is the page's own business when it was the page that asked.
        if (!on && view_) view_->ExecuteScript(L"if (document.fullscreenElement) document.exitFullscreen();", nullptr);
    }

    // ---- automation: run a script, take a picture ----

    void automate() {
        if (!opt_.eval_file.empty()) {
            std::string body;
            try {
                body = read_file_text(opt_.eval_file);
            } catch (const std::runtime_error& e) {
                fail(e.what(), kExitUsage);
                return;
            }
            run_script(wide(body), [this](bool ok, const std::string& json) {
                say(json + "\n");
                eval_ok_ = ok;
                if (opt_.screenshot.empty()) finish(ok ? 0 : 1);
                else after(static_cast<UINT>(opt_.wait_ms), [this] { capture(); });
            });
        } else {
            after(static_cast<UINT>(opt_.wait_ms), [this] { capture(); });
        }
    }

    // Runs `body` as the body of an async function in the page; `done` gets what it returned, as JSON.
    void run_script(const std::wstring& body, std::function<void(bool, const std::string&)> done) {
        // `__vitro_started` is set by the first statement of the script. A script with a syntax error is not run at all,
        // so the flag stays false and the poll below can say so at once instead of waiting for a result that never comes.
        view_->ExecuteScript(L"window.__vitro_eval = null; window.__vitro_started = false;", nullptr);
        const std::wstring wrapped =
            L"window.__vitro_started = true; (async () => { try { const v = await (async () => {\n" + body +
            L"\n})(); window.__vitro_eval = { ok: true, value: v === undefined ? null : v }; } catch (e) { "
            L"window.__vitro_eval = { ok: false, error: String((e && e.stack) || e) }; } })(); 0";
        view_->ExecuteScript(wrapped.c_str(), nullptr);
        poll_script(std::make_shared<std::function<void(bool, const std::string&)>>(std::move(done)));
    }

    void poll_script(std::shared_ptr<std::function<void(bool, const std::string&)>> done) {
        with_callback<ICoreWebView2ExecuteScriptCompletedHandler, HRESULT, LPCWSTR>(
            [this, done](HRESULT, LPCWSTR json) -> HRESULT {
                const std::string text = json ? utf8(json) : "null";
                if (text == "null") after(100, [this, done] { poll_script(done); });
                else (*done)(text.compare(0, 10, "{\"ok\":true") == 0, text);
                return S_OK;
            },
            [&](auto* cb) {
                return view_->ExecuteScript(
                    L"window.__vitro_eval || (window.__vitro_started ? null : "
                    L"{ ok: false, error: 'the script did not run: it has a syntax error' })",
                    cb);
            });
    }

    void capture() {
        ComPtr<IStream> stream;
        if (FAILED(CreateStreamOnHGlobal(nullptr, TRUE, stream.put()))) {
            fail("cannot make room for the picture", kExitFailed);
            return;
        }
        const HRESULT hr = with_callback<ICoreWebView2CapturePreviewCompletedHandler, HRESULT>(
            [this, stream](HRESULT done) -> HRESULT {
                if (FAILED(done)) {
                    fail("the window could not be captured", kExitFailed);
                    return S_OK;
                }
                finish(save_stream(stream.get(), opt_.screenshot) && eval_ok_ ? 0 : 1);
                return S_OK;
            },
            [&](auto* cb) { return view_->CapturePreview(COREWEBVIEW2_CAPTURE_PREVIEW_IMAGE_FORMAT_PNG, stream.get(), cb); });
        if (FAILED(hr)) fail("the window could not be captured", kExitFailed);
    }

    static bool save_stream(IStream* stream, const std::wstring& path) {
        HGLOBAL mem = nullptr;
        STATSTG st = {};
        if (FAILED(GetHGlobalFromStream(stream, &mem)) || FAILED(stream->Stat(&st, STATFLAG_NONAME))) {
            complain("cannot read the picture");
            return false;
        }
        const void* data = GlobalLock(mem);
        HANDLE f = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        bool ok = false;
        if (f != INVALID_HANDLE_VALUE) {
            DWORD n = 0;
            ok = WriteFile(f, data, static_cast<DWORD>(st.cbSize.QuadPart), &n, nullptr) && n == st.cbSize.QuadPart;
            CloseHandle(f);
        }
        GlobalUnlock(mem);
        if (!ok) complain("cannot write " + utf8(path) + " (Windows error " + std::to_string(GetLastError()) + ")");
        return ok;
    }

    Options opt_;
    Router router_;
    std::vector<Item> items_;
    HWND hwnd_ = nullptr;
    ComPtr<ICoreWebView2Environment> env_;
    ComPtr<ICoreWebView2Controller> controller_;
    ComPtr<ICoreWebView2> view_;
    std::map<UINT_PTR, std::function<void()>> timers_;
    UINT_PTR next_timer_ = 100;
    bool loaded_ = false;
    bool eval_ok_ = true;
    bool fullscreen_ = false;
    WINDOWPLACEMENT saved_ = {};
    int exit_code_ = 0;
};

std::string notices_text(const std::vector<Item>& items) {
    std::string out;
    for (const Item& it : items) {
        if (it.kind != "notice") continue;
        const Blob b = resource(it.id);
        out.append(reinterpret_cast<const char*>(b.data), b.size);
        out += "\n";
    }
    return out;
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    try {
        Options opt;
        try {
            opt = parse_command_line();
        } catch (const std::runtime_error& e) {
            complain(e.what());
            return kExitUsage;
        }
        if (opt.help) {
            say(kHelp);
            return 0;
        }
        std::vector<Item> items = table();
        if (opt.version) {
            std::string sdk;
            for (const Item& it : items) if (it.kind == "loader") sdk = it.name;
            say("vitro " VITRO_VERSION " (WebView2 SDK " + sdk + ")\n");
            return 0;
        }
        if (opt.licenses) {
            say(notices_text(items));
            return 0;
        }
        // The files given by name come first, then the built-in samples, as in `vitro web`; a file that is not a trace
        // stops the program before a window opens.
        std::vector<Entry> entries;
        for (const std::wstring& arg : opt.files) {
            wchar_t full[MAX_PATH * 4];
            const DWORD n = GetFullPathNameW(arg.c_str(), static_cast<DWORD>(sizeof full / sizeof full[0]), full, nullptr);
            if (n == 0 || GetFileAttributesW(full) == INVALID_FILE_ATTRIBUTES) {
                complain(utf8(arg) + ": no such file");
                return kExitUsage;
            }
            Entry e;
            e.path = full;
            e.name = utf8(e.path.substr(e.path.rfind(L'\\') + 1));
            e.shown = utf8(arg);
            e.source = "argument";
            bool seen = false;
            for (const Entry& other : entries) seen = seen || _wcsicmp(other.path.c_str(), e.path.c_str()) == 0;
            if (seen) continue;
            try {
                describe(e);
            } catch (const std::runtime_error& err) {
                complain(err.what());
                return kExitUsage;
            }
            entries.push_back(e);
        }
        for (const Item& it : items) {
            if (it.kind != "sample") continue;
            Entry e;
            e.name = it.name;
            e.shown = it.name;
            e.source = "sample";
            e.resource_id = it.id;
            entries.push_back(e);
        }
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        App app(opt, Router(std::move(entries), items), items);
        const int code = app.run(instance);
        CoUninitialize();
        return code;
    } catch (const std::exception& e) {
        complain(e.what());
        return kExitFailed;
    }
}
