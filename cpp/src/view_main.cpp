// vitro_view: Glass-box PyTorch trace viewer CLI and Windows application entry point.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "view_app.hpp"

namespace {

using vitro::VitroApp;

struct Options {
    std::string file, png, all, view = "tokens", events, library_dir, shots_dir, drop_test;
    bool digest = false, selftest = false, window_test = false, profile = false, keep_size = false;
    int flood = 0;
    int mouse_x = -1, mouse_y = -1;
    bool print_state = false;
    int unit = 0, token = -1, attn_fam = 0, layer = -1, head = -1, grid = 0, color = 0, net_k = 3, net_path = 0;
    float yaw = vitro::NET_YAW0, pitch = vitro::NET_PITCH0, zoom = 1.0f, phase = 0.0f;
    int width = 0, height = 0, bench = 0;
    float scale = 0.0f;
    bool set_unit = false, set_token = false, set_layer = false, set_head = false, set_fam = false;
};

int need_int(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("option ") + argv[i] + " needs a value");
    char* end = nullptr;
    long v = std::strtol(argv[i + 1], &end, 10);
    if (end == argv[i + 1] || *end) throw std::runtime_error(std::string("option ") + argv[i] + ": '" + argv[i + 1] + "' is not an integer");
    ++i;
    return static_cast<int>(v);
}

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
        else if (a == "--all") o.all = need_str(i, argc, argv);
        else if (a == "--view") o.view = need_str(i, argc, argv);
        else if (a == "--digest") o.digest = true;
        else if (a == "--selftest") o.selftest = true;
        else if (a == "--window-test") o.window_test = true;
        else if (a == "--profile") o.profile = true;
        else if (a == "--state") o.print_state = true;
        else if (a == "--events") o.events = need_str(i, argc, argv);
        else if (a == "--library-dir") o.library_dir = need_str(i, argc, argv);
        else if (a == "--shots-dir") o.shots_dir = need_str(i, argc, argv);
        else if (a == "--drop-test") o.drop_test = need_str(i, argc, argv);
        else if (a == "--test-keep-size") o.keep_size = true;
        else if (a == "--flood-test") o.flood = need_int(i, argc, argv);
        else if (a == "--unit") { o.unit = need_int(i, argc, argv); o.set_unit = true; }
        else if (a == "--token") { o.token = need_int(i, argc, argv); o.set_token = true; }
        else if (a == "--fam") { o.attn_fam = need_int(i, argc, argv); o.set_fam = true; }
        else if (a == "--layer") { o.layer = need_int(i, argc, argv); o.set_layer = true; }
        else if (a == "--head") { o.head = need_int(i, argc, argv); o.set_head = true; }
        else if (a == "--grid") o.grid = need_int(i, argc, argv);
        else if (a == "--color") o.color = need_int(i, argc, argv);
        else if (a == "--k") o.net_k = std::clamp(need_int(i, argc, argv), 1, 8);
        else if (a == "--path") o.net_path = need_int(i, argc, argv);
        else if (a == "--yaw") o.yaw = static_cast<float>(need_double(i, argc, argv));
        else if (a == "--pitch") o.pitch = static_cast<float>(need_double(i, argc, argv));
        else if (a == "--zoom") o.zoom = static_cast<float>(need_double(i, argc, argv));
        else if (a == "--phase") o.phase = static_cast<float>(need_double(i, argc, argv));
        else if (a == "--bench") o.bench = need_int(i, argc, argv);
        else if (a == "--scale") {
            o.scale = static_cast<float>(need_double(i, argc, argv));
            if (o.scale < 0.5f || o.scale > 4.0f) throw std::runtime_error("--scale between 0.5 and 4");
        } else if (a == "--size") {
            std::string v = need_str(i, argc, argv);
            if (std::sscanf(v.c_str(), "%dx%d", &o.width, &o.height) != 2 || o.width < 640 || o.height < 400)
                throw std::runtime_error("--size WIDTHxHEIGHT (>= 640x400), not '" + v + "'");
        } else if (a == "--mouse") {
            std::string v = need_str(i, argc, argv);
            if (std::sscanf(v.c_str(), "%d,%d", &o.mouse_x, &o.mouse_y) != 2)
                throw std::runtime_error("--mouse X,Y, not '" + v + "'");
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

vitro::View parse_view(const std::string& name) {
    for (int k = 0; k < vitro::VIEW_COUNT; ++k) {
        auto v = static_cast<vitro::View>(k);
        if (name == vitro::view_name(v)) return v;
    }
    throw std::runtime_error("unknown --view '" + name + "' (expected tokens|attention|activations|spans|net)");
}

void apply(VitroApp& app, const Options& o) {
    app.st.view = parse_view(o.view);
    if (o.set_unit) app.select_unit(o.unit);
    if (o.set_token) app.select_token(o.token);
    if (o.set_fam) app.st.attn_fam = o.attn_fam;
    if (o.set_layer) app.st.layer = o.layer;
    if (o.set_head) app.st.head = o.head;
    app.st.grid = o.grid != 0;
    app.st.color_mode = o.color;
    app.st.net_k = o.net_k;
    app.st.net_path = o.net_path != 0;
    app.st.net_yaw = o.yaw;
    app.st.net_pitch = o.pitch;
    app.st.net_zoom = o.zoom;
    app.st.net_time = static_cast<double>(o.phase) * 8.0;
    app.st.net_play = false;
    app.mx = o.mouse_x;
    app.my = o.mouse_y;
    app.clamp();
}

int digest(const vitro::TraceDump& d) {
    auto sum = [](const vitro::Arr& a) -> double {
        double s = 0.0;
        for (size_t i = 0; i < a.n; ++i) {
            float v = a.f(i);
            if (!std::isnan(v)) s += v;
        }
        return s;
    };
    std::printf("{\"uid\":\"%s\",\"labels\":%d,\"spans\":%d,\"units\":[",
                vitro::json_escape(d.uid).c_str(), static_cast<int>(d.labels.size()), static_cast<int>(d.spans.size()));
    for (size_t i = 0; i < d.units.size(); ++i) {
        const vitro::UnitTrace& u = d.units[i];
        if (i) std::printf(",");
        double sum_w = 0.0;
        for (const auto& lk : u.links) sum_w += sum(lk.weights);
        double sum_act = 0.0;
        for (const auto& st : u.stages) sum_act += sum(st.act);
        std::printf("{\"u\":%d,\"n\":%d,\"free_n\":%d,\"diverge\":%d,\"sum_tok\":%.17g,\"sum_p\":%.17g,\"sum_rank\":%.17g,"
                    "\"sum_top_p\":%.17g,\"sum_entropy\":%.17g,\"sum_weights\":%.17g,\"sum_act\":%.17g,\"sum_norms\":%.17g}",
                    u.u, u.n_tokens(), u.free_n(), u.diverge, sum(u.tok), sum(u.p), sum(u.rank), sum(u.top_p), sum(u.entropy),
                    sum_w, sum_act, sum(u.norms.data));
    }
    std::printf("]}\n");
    return 0;
}

int run_png(VitroApp& app, const Options& o) {
    if (!o.events.empty()) vitro::play_events_report(app, o.events);
    app.render();
    app.cv.save_png(o.png);
    if (o.print_state) std::printf("state: %s\n", app.state_json().c_str());
    return 0;
}

int run_all(VitroApp& app, const Options& o) {
    CreateDirectoryA(o.all.c_str(), nullptr);
    for (int k = 0; k < vitro::VIEW_COUNT; ++k) {
        Options oo = o;
        oo.view = vitro::view_name(static_cast<vitro::View>(k));
        apply(app, oo);
        app.render();
        std::string path = o.all + "/" + oo.view + ".png";
        app.cv.save_png(path);
        std::printf("%s\n", path.c_str());
    }
    return 0;
}

std::string default_library() {
    const char* env = std::getenv("VITRO_DUMP_DIR");
    if (env && *env) return env;
    std::string beside = vitro::exe_dir() + "/traces";
    if (vitro::dir_exists(beside)) return beside;
    return ".";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        if (o.selftest) return vitro::run_selftest();
        if (o.digest) {
            if (o.file.empty()) throw std::runtime_error("--digest needs a file");
            vitro::TraceDump d(o.file);
            return digest(d);
        }
        if (!o.all.empty() && o.file.empty()) throw std::runtime_error("--all needs a trace file");
        const bool window = o.png.empty() && o.all.empty();
        const float scale = o.scale > 0 ? o.scale : (window ? vitro::display_scale() : 1.0f);
        const int w = o.width ? o.width : (window ? static_cast<int>(1500 * scale) : 1700);
        const int h = o.height ? o.height : (window ? static_cast<int>(940 * scale) : 1000);
        VitroApp app(w, h, scale);
        app.adaptive = window;
        app.library_dir = o.library_dir.empty() ? default_library() : o.library_dir;
        if (window && !o.window_test) {
            app.recent_file = vitro::exe_dir() + "/vitro_view.recent.txt";
            app.load_recent();
        }
        if (!o.file.empty()) {
            std::string why = app.open_file(o.file);
            if (!why.empty() && !window) throw std::runtime_error(why);
            if (why.empty()) apply(app, o);
        }
        if (!o.all.empty()) return run_all(app, o);
        if (!o.png.empty()) return run_png(app, o);
        vitro::ShellConfig cfg;
        cfg.window_class = L"VitroView";
        cfg.extension = "vtdump";
        cfg.filter_name = "Vitro Trace Dumps";
        cfg.app_name = "vitro_view";
        cfg.shots_dir = o.shots_dir.empty() ? app.library_dir + "/shots" : o.shots_dir;
        cfg.width = w;
        cfg.height = h;
        cfg.min_width = static_cast<int>(1200 * scale);
        cfg.min_height = static_cast<int>(680 * scale);
        cfg.quit_on_escape = true;
        cfg.window_test = o.window_test;
        cfg.test_drop_file = o.drop_test;
        cfg.test_keep_size = o.keep_size;
        cfg.test_flood = o.flood;
        return vitro::run_shell(app, cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vitro_view: %s\n", e.what());
        return 1;
    }
}
