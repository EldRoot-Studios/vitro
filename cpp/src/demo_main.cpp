// vitro_demo: 3D neural network generation step player CLI and Windows entry point.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "demo_app.hpp"

namespace {

using vitro::DemoApp;

struct Options {
    std::string file, png, events, library_dir, shots_dir, drop_test;
    bool digest = false, state = false, window_test = false, keep_size = false;
    int flood = 0, seg = -1, token = -1, width = 0, height = 0;
    double tick = -1, read = -1, frac = 0.5, speed = 1.0;
    float yaw = 0.42f, pitch = 0.16f, zoom = 1.0f, scale = 0.0f;
    bool set_camera = false;
};

double need_double(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("option ") + argv[i] + " needs a value");
    char* end = nullptr;
    double v = std::strtod(argv[i + 1], &end);
    if (end == argv[i + 1] || *end) throw std::runtime_error(std::string("option ") + argv[i] + ": '" + argv[i + 1] + "' is not a number");
    ++i;
    return v;
}

std::string need_str(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("option ") + argv[i] + " needs a value");
    return argv[++i];
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--png") o.png = need_str(i, argc, argv);
        else if (a == "--digest") o.digest = true;
        else if (a == "--state") o.state = true;
        else if (a == "--window-test") o.window_test = true;
        else if (a == "--events") o.events = need_str(i, argc, argv);
        else if (a == "--library-dir") o.library_dir = need_str(i, argc, argv);
        else if (a == "--shots-dir") o.shots_dir = need_str(i, argc, argv);
        else if (a == "--drop-test") o.drop_test = need_str(i, argc, argv);
        else if (a == "--test-keep-size") o.keep_size = true;
        else if (a == "--flood-test") o.flood = static_cast<int>(need_double(i, argc, argv));
        else if (a == "--seg") o.seg = static_cast<int>(need_double(i, argc, argv));
        else if (a == "--token") o.token = static_cast<int>(need_double(i, argc, argv));
        else if (a == "--tick") o.tick = need_double(i, argc, argv);
        else if (a == "--read") o.read = need_double(i, argc, argv);
        else if (a == "--frac") o.frac = need_double(i, argc, argv);
        else if (a == "--speed") o.speed = need_double(i, argc, argv);
        else if (a == "--yaw") { o.yaw = static_cast<float>(need_double(i, argc, argv)); o.set_camera = true; }
        else if (a == "--pitch") { o.pitch = static_cast<float>(need_double(i, argc, argv)); o.set_camera = true; }
        else if (a == "--zoom") { o.zoom = static_cast<float>(need_double(i, argc, argv)); o.set_camera = true; }
        else if (a == "--scale") {
            o.scale = static_cast<float>(need_double(i, argc, argv));
            if (o.scale < 0.5f || o.scale > 4.0f) throw std::runtime_error("--scale between 0.5 and 4");
        } else if (a == "--size") {
            std::string v = need_str(i, argc, argv);
            if (std::sscanf(v.c_str(), "%dx%d", &o.width, &o.height) != 2 || o.width < 640 || o.height < 400)
                throw std::runtime_error("--size WIDTHxHEIGHT (>= 640x400), not '" + v + "'");
        } else if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
            throw std::runtime_error("unknown option " + a);
        } else if (o.file.empty()) {
            o.file = a;
        } else {
            throw std::runtime_error("unexpected argument " + a);
        }
    }
    return o;
}

int digest(const vitro::TraceDemo& d) {
    auto sum = [](const vitro::Arr& a) -> double {
        double s = 0.0;
        for (size_t i = 0; i < a.n; ++i) {
            float v = a.f(i);
            if (!std::isnan(v)) s += v;
        }
        return s;
    };
    std::printf("{\"uid\":\"%s\",\"total\":%.6f,\"segments\":[", vitro::json_escape(d.uid).c_str(), d.total);
    for (size_t i = 0; i < d.segments.size(); ++i) {
        const vitro::DemoSegment& sg = d.segments[i];
        if (i) std::printf(",");
        std::printf("{\"k\":%d,\"m\":%d,\"sum_ftok\":%.17g,\"sum_fp\":%.17g,\"sum_top_p\":%.17g,\"sum_self_w\":%.17g,\"sum_norm\":%.17g}",
                    sg.k, sg.free_n(), sum(sg.ftok), sum(sg.fp), sum(sg.tape.top_p), sum(sg.tape.self_w), sum(sg.tape.norm));
    }
    std::printf("]}\n");
    return 0;
}

std::string default_library() {
    const char* env = std::getenv("VITRO_DEMO_DIR");
    if (env && *env) return env;
    std::string beside = vitro::exe_dir() + "/demos";
    if (vitro::dir_exists(beside)) return beside;
    return ".";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        if (o.digest) {
            if (o.file.empty()) throw std::runtime_error("--digest needs a file");
            vitro::TraceDemo d(o.file);
            return digest(d);
        }
        const bool window = o.png.empty();
        const float scale = o.scale > 0 ? o.scale : (window ? vitro::display_scale() : 1.0f);
        const int w = o.width ? o.width : (window ? static_cast<int>(1460 * scale) : 1600);
        const int h = o.height ? o.height : (window ? static_cast<int>(900 * scale) : 960);
        DemoApp app(w, h, scale);
        app.speed = o.speed;
        if (o.set_camera) { app.yaw = o.yaw; app.pitch = o.pitch; app.cam_zoom = o.zoom; }
        app.library_dir = o.library_dir.empty() ? default_library() : o.library_dir;
        if (window && !o.window_test) { app.recent_file = vitro::exe_dir() + "/vitro_demo.recent.txt"; app.load_recent(); }
        if (!o.file.empty()) {
            std::string why = app.open_file(o.file);
            if (!why.empty() && !window) throw std::runtime_error(why);
            if (why.empty()) {
                app.playing = window;
                if (o.tick >= 0.0) app.pos = std::clamp(o.tick, 0.0, std::max(0.0, app.demo.total - 1e-6));
                else if (o.read >= 0.0) {
                    int si = std::clamp(o.seg >= 0 ? o.seg : 0, 0, static_cast<int>(app.demo.segments.size()) - 1);
                    const auto& sg = app.demo.segments[static_cast<size_t>(si)];
                    app.pos = sg.start + std::clamp(o.read, 0.0, 0.999) * std::max(0.0, sg.read_ticks);
                } else if (o.token >= 0) {
                    int si = std::clamp(o.seg >= 0 ? o.seg : 0, 0, static_cast<int>(app.demo.segments.size()) - 1);
                    const auto& sg = app.demo.segments[static_cast<size_t>(si)];
                    app.pos = sg.start + sg.read_ticks + std::max(0, o.token - sg.t0) + std::clamp(o.frac, 0.0, 0.999);
                }
            }
        }
        if (!o.png.empty()) {
            if (!o.events.empty()) vitro::play_events_report(app, o.events);
            app.render();
            app.cv.save_png(o.png);
            if (o.state) std::printf("state: %s\n", app.state_json().c_str());
            return 0;
        }
        vitro::ShellConfig cfg;
        cfg.window_class = L"VitroDemo";
        cfg.extension = "vtdemo";
        cfg.filter_name = "Vitro Demo Recordings";
        cfg.app_name = "vitro_demo";
        cfg.shots_dir = o.shots_dir.empty() ? app.library_dir + "/shots" : o.shots_dir;
        cfg.width = w;
        cfg.height = h;
        cfg.min_width = static_cast<int>(1080 * scale);
        cfg.min_height = static_cast<int>(640 * scale);
        cfg.quit_on_escape = true;
        cfg.window_test = o.window_test;
        cfg.test_drop_file = o.drop_test;
        cfg.test_keep_size = o.keep_size;
        cfg.test_flood = o.flood;
        return vitro::run_shell(app, cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vitro_demo: %s\n", e.what());
        return 1;
    }
}
