// Vitro 3D generation player implementation: timeline playback, dynamic 3D neural rays, token strip, spans, and library.
#include "demo_app.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <ctime>
#include <numeric>

namespace vitro {

std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

namespace {

const double SPEEDS[] = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0};
constexpr int N_SPEEDS = 7;
constexpr int TRAIL = 5;

Color violet() { return rgb(0x7b6df0); }
Color gold() { return rgb(0xffc94a); }
Color magenta() { return rgb(0xb06cf0); }
Color cyan_col() { return rgb(0x4fb0c6); }

}  // namespace

DemoApp::Now DemoApp::now() const {
    Now n;
    if (!demo.loaded()) return n;
    double p = std::clamp(pos, 0.0, std::max(0.0, demo.total - 1e-9));
    n.seg = demo.segment_at(p);
    const DemoSegment& sg = demo.segments[static_cast<size_t>(n.seg)];
    double local = p - sg.start;
    if (sg.read_ticks > 0.0 && local < sg.read_ticks) {
        n.reading = true;
        n.read_p = local / sg.read_ticks;
        n.token = sg.t0;
        return n;
    }
    double w = local - sg.read_ticks;
    int max_steps = std::max(1, sg.free_n() - sg.t0);
    int i = std::clamp(static_cast<int>(std::floor(w)), 0, max_steps - 1);
    n.reading = false;
    n.read_p = 1.0;
    n.token = sg.t0 + i;
    n.frac = std::clamp(w - i, 0.0, 1.0);
    return n;
}

void DemoApp::seek_segment(int seg) {
    if (seg < 0 || seg >= static_cast<int>(demo.segments.size())) return;
    pos = demo.segments[static_cast<size_t>(seg)].start;
}

void DemoApp::seek_token(int seg, int token) {
    if (seg < 0 || seg >= static_cast<int>(demo.segments.size())) return;
    const DemoSegment& sg = demo.segments[static_cast<size_t>(seg)];
    int tok = std::clamp(token, sg.t0, sg.free_n() - 1);
    pos = sg.start + sg.read_ticks + (tok - sg.t0) + 0.5;
}

void DemoApp::toggle_play() {
    if (!playing && pos >= demo.total - 1.0 && !loop) pos = 0.0;
    playing = !playing;
}

void DemoApp::step(int by) {
    playing = false;
    Now n = now();
    const DemoSegment& sg = demo.segments[static_cast<size_t>(n.seg)];
    double at = n.reading ? sg.start + sg.read_ticks : sg.start + sg.read_ticks + (n.token - sg.t0) + 0.5;
    pos = std::clamp(at + by, 0.0, std::max(0.0, demo.total - 1e-6));
}

void DemoApp::jump_segment(int dir) {
    Now n = now();
    int target = std::clamp(n.seg + dir, 0, static_cast<int>(demo.segments.size()) - 1);
    seek_segment(target);
}

bool DemoApp::tick(double seconds) {
    bool changed = false;
    if (notice_left_ > 0.0) {
        notice_left_ -= seconds;
        if (notice_left_ <= 0.0) { notice_text_.clear(); changed = true; }
    }
    if (!demo.loaded() || !playing) return changed;
    sway_ += seconds;
    pos += seconds * TICKS_PER_SECOND * speed;
    if (pos >= demo.total) {
        if (loop) pos = std::fmod(pos, std::max(1e-6, demo.total));
        else { pos = std::max(0.0, demo.total - 1e-6); playing = false; }
    }
    return true;
}

std::string DemoApp::open_file(const std::string& path) {
    try {
        TraceDemo fresh(path);
        demo = std::move(fresh);
    } catch (const std::exception& e) {
        error_ = e.what();
        status_ = error_;
        return error_;
    }
    pos = 0.0;
    playing = true;
    scene_.reset();
    seg_norm_max_.assign(demo.segments.size(), -1.0f);
    library_ = false;
    library_scroll_ = 0;
    error_.clear();
    remember(path);
    status_ = "Opened " + file_name_of(path);
    return "";
}

bool DemoApp::escape() {
    if (help_) { help_ = false; return true; }
    if (library_ && demo.loaded()) { library_ = false; return true; }
    return false;
}

std::string DemoApp::title() {
    return demo.loaded() ? ("Vitro Demo - " + demo.title) : "Vitro Demo Player";
}

void DemoApp::set_ui_zoom(float zoom) {
    zoom_ = std::clamp(zoom, 0.6f, 3.0f);
    cv.set_scale(base_scale_ * zoom_);
}

void DemoApp::load_recent() { recent_ = read_recent(recent_file); library_dirty_ = true; }

void DemoApp::remember(std::string path) {
    recent_.erase(std::remove(recent_.begin(), recent_.end(), path), recent_.end());
    recent_.insert(recent_.begin(), path);
    if (recent_.size() > 12) recent_.resize(12);
    library_dirty_ = true;
    write_recent(recent_file, recent_);
}

void DemoApp::scan_library() {
    library_items_ = vitro::scan_library(library_dir, "vtdemo", recent_);
    library_dirty_ = false;
}

const DemoHit* DemoApp::hit_at(int x, int y) const {
    for (size_t i = hits_.size(); i-- > 0;)
        if (hits_[i].r.contains(x, y)) return &hits_[i];
    return nullptr;
}

bool DemoApp::button(Rect r, const std::string& label, bool active, int kind, int a, FontId f) {
    bool hov = r.contains(mx_, my_);
    cv.fill(r, active ? rgb(0x2f5f9e) : (hov ? rgb(0x39404a) : pal::panel2));
    cv.frame(r, active ? pal::accent : pal::border);
    cv.text_fit(r.x + s(3), r.y + (r.h - cv.fonts.height(f)) / 2, std::max(1, r.w - s(6)), label, active ? rgb(0xffffff) : pal::text, f, 1);
    add_hit(r, kind, a);
    return hov;
}

void DemoApp::scrub_to(int x) {
    if (timeline_rect_.w <= 0 || !demo.loaded()) return;
    double f = std::clamp(static_cast<double>(x - timeline_rect_.x) / timeline_rect_.w, 0.0, 1.0);
    pos = f * std::max(0.0, demo.total - 1e-6);
}

void DemoApp::move(int x, int y) {
    int dx = x - mx_, dy = y - my_;
    bool had = mx_ >= 0;
    mx_ = x; my_ = y;
    if (!dragging_) return;
    if (std::abs(x - press_x_) + std::abs(y - press_y_) > s(4)) moved_ = true;
    if (scrub_) scrub_to(x);
    else if (scrolling_) library_scroll_ = scrollbar_offset_at(cv, lib_sv_, y);
    else if (rotating_ && moved_ && had) {
        yaw += static_cast<float>(dx) * 0.008f;
        pitch = std::clamp(pitch + static_cast<float>(dy) * 0.008f, -1.45f, 1.45f);
    }
}

void DemoApp::press(int x, int y) {
    mx_ = x; my_ = y;
    dragging_ = true; moved_ = false;
    press_x_ = x; press_y_ = y;
    scrub_ = rotating_ = scrolling_ = false;
    if (!demo.loaded() || library_) {
        const DemoHit* h = hit_at(x, y);
        if (h && h->kind == DH_SCROLL) { scrolling_ = true; library_scroll_ = scrollbar_offset_at(cv, lib_sv_, y); }
        return;
    }
    if (timeline_rect_.contains(x, y)) { scrub_ = true; scrub_to(x); }
    else if (scene_rect_.contains(x, y)) { const DemoHit* h = hit_at(x, y); rotating_ = !h || h->kind == DH_SCENE; }
}

void DemoApp::release(int x, int y) {
    mx_ = x; my_ = y;
    bool was_click = dragging_ && !moved_ && !scrub_ && !scrolling_;
    dragging_ = scrub_ = rotating_ = scrolling_ = false;
    moved_ = false;
    if (was_click) click(x, y);
}

void DemoApp::wheel(int x, int y, int delta, bool ctrl) {
    (void)ctrl;
    if (!demo.loaded() || library_) { library_scroll_ = std::max(0, library_scroll_ - (delta > 0 ? 1 : -1) * s(78)); return; }
    if (scene_rect_.contains(x, y)) cam_zoom = std::clamp(cam_zoom * (delta > 0 ? 1.1f : 1.0f / 1.1f), 0.3f, 5.0f);
}

void DemoApp::click(int x, int y) {
    const DemoHit* found = hit_at(x, y);
    if (!found) return;
    DemoHit h = *found;
    switch (h.kind) {
        case DH_PLAY: toggle_play(); break;
        case DH_STEP_BACK: step(-1); break;
        case DH_STEP_FWD: step(1); break;
        case DH_PREV_SEG: jump_segment(-1); break;
        case DH_NEXT_SEG: jump_segment(1); break;
        case DH_SPEED: speed = SPEEDS[std::clamp(h.a, 0, N_SPEEDS - 1)]; break;
        case DH_LOOP: loop = !loop; break;
        case DH_GHOST: ghost = !ghost; break;
        case DH_SIDE_SEG: seek_segment(h.a); break;
        case DH_SPAN_JUMP: seek_token(now().seg, h.a); break;
        case DH_LIB_ITEM: open_file(library_items_[static_cast<size_t>(h.a)].path); break;
        case DH_LIB_OPEN: request_ = CMD_OPEN; break;
        case DH_TIMELINE: scrub_to(x); break;
        default: break;
    }
}

bool DemoApp::key(int vk, bool ctrl, bool shift) {
    (void)ctrl;
    if (!demo.loaded() || library_) {
        const int page = std::max(s(100), lib_sv_.area.h * 9 / 10);
        switch (vk) {
            case 0x70: help_ = !help_; return true;
            case 'B': if (!library_) return false; library_ = false; return true;
            case 0x21: library_scroll_ = std::max(0, library_scroll_ - page); return true;
            case 0x22: library_scroll_ += page; return true;
            case 0x24: library_scroll_ = 0; return true;
            case 0x23: library_scroll_ = 1 << 20; return true;
            default: return false;
        }
    }
    auto speed_index = [&]() {
        int best = 0;
        for (int i = 0; i < N_SPEEDS; ++i)
            if (std::fabs(SPEEDS[i] - speed) < std::fabs(SPEEDS[best] - speed)) best = i;
        return best;
    };
    switch (vk) {
        case 0x20: toggle_play(); return true;
        case 0x25: step(shift ? -10 : -1); return true;
        case 0x27: step(shift ? 10 : 1); return true;
        case 0x26: speed = SPEEDS[std::min(N_SPEEDS - 1, speed_index() + 1)]; return true;
        case 0x28: speed = SPEEDS[std::max(0, speed_index() - 1)]; return true;
        case 0xDB: jump_segment(-1); return true;
        case 0xDD: jump_segment(1); return true;
        case 0x24: pos = 0.0; return true;
        case 0x23: pos = std::max(0.0, demo.total - 1e-6); playing = false; return true;
        case 'L': loop = !loop; return true;
        case 'G': ghost = !ghost; return true;
        case 'B': library_ = true; library_dirty_ = true; return true;
        case 'R': yaw = 0.42f; pitch = 0.16f; cam_zoom = 1.0f; return true;
        case 'D': {
            Now n = now();
            const DemoSegment& sg = demo.segments[static_cast<size_t>(n.seg)];
            if (sg.diverge >= sg.t0 && sg.diverge < sg.free_n()) {
                seek_token(n.seg, sg.diverge);
                playing = false;
                return true;
            }
            status_ = sg.diverge < 0 ? "No divergence against reference in this segment" : "Divergence lies outside generated length";
            return false;
        }
        case 0x70: help_ = !help_; return true;
        default: return false;
    }
}

enum DemoCommand {
    C_LIBRARY = CMD_APP_FIRST + 1, C_PLAY, C_HOME, C_PREV_SEG, C_NEXT_SEG, C_BACK, C_FWD, C_LOOP, C_GHOST, C_DIVERGE, C_CAMERA, C_HELP,
    C_SPEED0 = CMD_APP_FIRST + 40
};

std::vector<Menu> DemoApp::menus() {
    std::vector<Menu> m;
    m.push_back({"File", {{CMD_OPEN, "Open Demo...\tCtrl+O", false}, {C_LIBRARY, "Library\tB", false}, {CMD_SAVE_PNG, "Save PNG...\tCtrl+S", false}, {0, "", false}, {CMD_EXIT, "Exit\tQ", false}}});
    m.push_back({"Playback", {{C_PLAY, playing ? "Pause\tSpace" : "Play\tSpace", false}, {C_HOME, "Start\tHome", false}, {C_PREV_SEG, "Previous Segment\t[", false},
                              {C_NEXT_SEG, "Next Segment\t]", false}, {C_BACK, "Step Back\tLeft", false}, {C_FWD, "Step Forward\tRight", false}, {0, "", false},
                              {C_LOOP, "Loop\tL", loop}, {C_GHOST, "Show Reference Overlay\tG", ghost}, {C_DIVERGE, "Jump to Divergence\tD", false}}});
    Menu sp{"Speed", {}};
    static const char* names[N_SPEEDS] = {"0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x"};
    for (int i = 0; i < N_SPEEDS; ++i) sp.items.push_back({C_SPEED0 + i, names[i], std::fabs(SPEEDS[i] - speed) < 1e-9});
    m.push_back(sp);
    m.push_back({"View", {{CMD_ZOOM_IN, "Zoom In\tCtrl++", false}, {CMD_ZOOM_OUT, "Zoom Out\tCtrl+-", false}, {CMD_ZOOM_RESET, "Reset Zoom\tCtrl+0", false},
                          {CMD_FULLSCREEN, "Fullscreen\tF11", false}, {0, "", false}, {C_CAMERA, "Reset 3D Camera\tR", false}}});
    m.push_back({"Help", {{C_HELP, "Controls\tF1", help_}}});
    return m;
}

void DemoApp::command(int id) {
    if (id >= C_SPEED0 && id < C_SPEED0 + N_SPEEDS) { speed = SPEEDS[id - C_SPEED0]; return; }
    switch (id) {
        case C_LIBRARY: library_ = true; library_dirty_ = true; break;
        case C_PLAY: if (demo.loaded()) toggle_play(); break;
        case C_HOME: key(0x24, false, false); break;
        case C_PREV_SEG: key(0xDB, false, false); break;
        case C_NEXT_SEG: key(0xDD, false, false); break;
        case C_BACK: key(0x25, false, false); break;
        case C_FWD: key(0x27, false, false); break;
        case C_LOOP: loop = !loop; break;
        case C_GHOST: ghost = !ghost; break;
        case C_DIVERGE: key('D', false, false); break;
        case C_CAMERA: key('R', false, false); break;
        case C_HELP: help_ = !help_; break;
        default: break;
    }
}

std::string DemoApp::state_json() {
    char buf[900];
    if (!demo.loaded()) {
        std::snprintf(buf, sizeof buf, "{\"loaded\":false,\"library\":%d,\"help\":%d,\"zoom\":%.3f,\"recent\":%d,\"error\":\"",
                      library_ ? 1 : 0, help_ ? 1 : 0, zoom_, static_cast<int>(recent_.size()));
    } else {
        Now n = now();
        std::snprintf(buf, sizeof buf,
                      "{\"loaded\":true,\"uid\":\"%s\",\"pos\":%.4f,\"total\":%.2f,\"playing\":%d,\"speed\":%.3f,\"loop\":%d,\"ghost\":%d,\"seg\":%d,\"segments\":%d,"
                      "\"phase\":\"%s\",\"token\":%d,\"zoom\":%.3f,\"yaw\":%.4f,\"pitch\":%.4f,\"cam_zoom\":%.4f,\"library\":%d,\"help\":%d,\"error\":\"",
                      json_escape(demo.uid).c_str(), pos, demo.total, playing ? 1 : 0, speed, loop ? 1 : 0, ghost ? 1 : 0,
                      n.seg, static_cast<int>(demo.segments.size()), n.reading ? "read" : "write", n.token,
                      zoom_, yaw, pitch, cam_zoom, library_ ? 1 : 0, help_ ? 1 : 0);
    }
    return std::string(buf) + json_escape(error_) + "\",\"status\":\"" + json_escape(status_) + "\"}";
}

float DemoApp::norm_max(int seg) {
    float& v = seg_norm_max_[static_cast<size_t>(seg)];
    if (v <= 0.0f) {
        const DemoTape& t = demo.segments[static_cast<size_t>(seg)].tape;
        float mx = 1e-6f;
        const size_t C = t.norm.dim(2);
        for (size_t i = 0; i < t.norm.dim(0); ++i)
            for (size_t l = 0; l < t.norm.dim(1); ++l) mx = std::max(mx, t.norm.f((i * t.norm.dim(1) + l) * C));
        v = mx;
    }
    return v;
}

std::shared_ptr<DemoScene> DemoApp::scene_for(int seg) {
    if (scene_ && scene_->seg == seg) return scene_;
    auto sc = std::make_shared<DemoScene>();
    DemoScene& n = *sc;
    n.seg = seg;
    const DemoSegment& sg = demo.segments[static_cast<size_t>(seg)];
    const int ns = static_cast<int>(sg.stages.size());
    n.stage_x.assign(static_cast<size_t>(ns), 0.0f);
    float cur_x = 0.0f;
    for (int i = 0; i < ns; ++i) {
        if (i > 0) cur_x += 40.0f + sg.stages[static_cast<size_t>(i)].gap_before;
        n.stage_x[static_cast<size_t>(i)] = cur_x;
    }
    n.half_span = cur_x * 0.5f;
    n.first.assign(static_cast<size_t>(ns), 0);
    n.count.assign(static_cast<size_t>(ns), 0);

    for (int st_idx = 0; st_idx < ns; ++st_idx) {
        const StageDef& st = sg.stages[static_cast<size_t>(st_idx)];
        n.first[static_cast<size_t>(st_idx)] = static_cast<int>(n.nodes.size());
        n.count[static_cast<size_t>(st_idx)] = st.count;
        float amax = 1e-6f;
        for (size_t i = 0; i < st.act.n; ++i) amax = std::max(amax, st.act.f(i));
        for (int i = 0; i < st.count; ++i) {
            DSceneNode nd;
            nd.stage = st_idx; nd.idx = i;
            nd.x = n.stage_x[static_cast<size_t>(st_idx)] + (hash01(static_cast<uint32_t>(st_idx), static_cast<uint32_t>(i) + 9000u) - 0.5f) * st.x_jitter;
            nd.y = (0.5f - (static_cast<float>(i) + 0.5f) / static_cast<float>(std::max(1, st.count))) * 2.0f * st.span_y;
            nd.z = (hash01(static_cast<uint32_t>(st_idx), static_cast<uint32_t>(i)) - 0.5f) * 2.0f * st.z_jitter;
            nd.act = static_cast<size_t>(i) < st.act.n ? std::clamp(st.act.f(static_cast<size_t>(i)) / amax, 0.1f, 1.0f) : 0.25f;
            n.nodes.push_back(nd);
        }
    }
    for (int st_idx = 0; st_idx < ns;) {
        const std::string& grp = sg.stages[static_cast<size_t>(st_idx)].group;
        int e = st_idx;
        while (e + 1 < ns && sg.stages[static_cast<size_t>(e + 1)].group == grp) ++e;
        n.groups.push_back({st_idx, e, grp.empty() ? sg.stages[static_cast<size_t>(st_idx)].label : grp});
        st_idx = e + 1;
    }

    for (const PrefillLink& pl : sg.prefill_links) {
        const int dst_cnt = static_cast<int>(pl.ids.dim(0));
        const int K = static_cast<int>(pl.ids.dim(1));
        const Color col = parse_tint(pl.tint, pl.kind == "cross" ? cyan_col() : violet());
        for (int b = 0; b < dst_cnt && b < n.count[static_cast<size_t>(pl.dst)]; ++b)
            for (int kk = 0; kk < K; ++kk) {
                int a_idx = pl.ids.iv(static_cast<size_t>(b) * K + kk);
                float w = pl.weights.f(static_cast<size_t>(b) * K + kk);
                if (a_idx >= 0 && a_idx < n.count[static_cast<size_t>(pl.src)] && w > 0.02f)
                    n.edges.push_back({n.node(pl.src, a_idx), n.node(pl.dst, b), w, pl.dst, col});
            }
    }
    scene_ = sc;
    return sc;
}

void DemoApp::draw_scene(Rect view) {
    scene_rect_ = view;
    cv.fill(view, rgb(0x0b0a1a));
    cv.frame(view, pal::border);
    Now nw = now();
    std::shared_ptr<DemoScene> sp = scene_for(nw.seg);
    DemoScene& n = *sp;
    const DemoSegment& sg = demo.segments[static_cast<size_t>(nw.seg)];
    const int Ld = static_cast<int>(sg.layer_stages.size());
    const int Kself = static_cast<int>(sg.tape.self_id.dim(2));
    const int Kcross = sg.cross_stage >= 0 ? static_cast<int>(sg.tape.cross_id.dim(2)) : 0;
    const int Ccross = sg.cross_stage >= 0 ? n.count[static_cast<size_t>(sg.cross_stage)] : 0;
    const int t = nw.reading ? sg.t0 : nw.token;

    Camera3D c;
    float yaw_eff = yaw + (playing ? 0.16f * std::sin(static_cast<float>(sway_) * 0.45f) : 0.0f);
    c.cy = std::cos(yaw_eff); c.sy = std::sin(yaw_eff);
    c.cp = std::cos(pitch); c.sp = std::sin(pitch);
    c.half_span = n.half_span;
    c.dist = std::max(120.0f, n.half_span * 3.4f);
    c.vx = static_cast<float>(view.x + view.w / 2);
    c.vy = static_cast<float>(view.y + view.h / 2 + s(6));
    c.ox = c.vx; c.oy = c.vy; c.focal = c.dist;
    auto project_all = [&]() { for (DSceneNode& nd : n.nodes) c.map(nd.x, nd.y, nd.z, &nd.sx, &nd.sy, &nd.scale); };
    project_all();
    float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
    for (const DSceneNode& nd : n.nodes) {
        minx = std::min(minx, nd.sx); maxx = std::max(maxx, nd.sx);
        miny = std::min(miny, nd.sy); maxy = std::max(maxy, nd.sy);
    }
    float fit = std::min(0.92f * static_cast<float>(view.w) / std::max(1.0f, maxx - minx),
                         0.70f * static_cast<float>(view.h) / std::max(1.0f, maxy - miny));
    c.focal *= fit;
    c.ox = c.vx - ((minx + maxx) / 2 - c.vx) * fit;
    c.oy = c.vy - ((miny + maxy) / 2 - c.vy) * fit;
    c.zoom = cam_zoom;
    project_all();

    cv.push_clip(view);
    const double wave = nw.reading ? nw.read_p * static_cast<double>(std::max(1, sg.in_stage)) : -10.0;
    for (const DSceneEdge& e : n.edges) {
        const DSceneNode& p = n.nodes[static_cast<size_t>(e.a)];
        const DSceneNode& q = n.nodes[static_cast<size_t>(e.b)];
        float boost = nw.reading ? (0.45f + 2.4f * static_cast<float>(std::exp(-std::pow(static_cast<double>(e.stage) - 0.5 - wave, 2.0) / 1.3))) : 0.7f;
        cv.line_add(p.sx, p.sy, q.sx, q.sy, e.tint, std::min(0.9f, (0.03f + 0.14f * std::sqrt(std::min(1.0f, e.w))) * boost));
    }

    std::vector<float> cross_use(static_cast<size_t>(Ccross), 0.0f);
    if (!nw.reading && sg.cross_stage >= 0 && Kcross > 0) {
        for (int l = 0; l < Ld; ++l)
            for (int kk = 0; kk < Kcross; ++kk) {
                size_t at = (static_cast<size_t>(t) * Ld + l) * Kcross + kk;
                int id = sg.tape.cross_id.iv(at);
                float w = sg.tape.cross_w.f(at);
                if (id >= 0 && id < Ccross) cross_use[static_cast<size_t>(id)] += w / static_cast<float>(Ld);
            }
    }
    float cuse_max = 1e-6f;
    for (float v : cross_use) cuse_max = std::max(cuse_max, v);

    if (!nw.reading) {
        const int Pn = n.count[static_cast<size_t>(sg.in_stage)];
        for (int age = TRAIL; age >= 0; --age) {
            int tt = t - age;
            if (tt < sg.t0 || tt - 1 >= Pn) continue;
            float fade = std::pow(0.55f, static_cast<float>(age));
            const int q = tt - 1;
            for (int l = 0; l < Ld; ++l) {
                float reveal = age == 0 ? std::clamp(static_cast<float>(nw.frac) * static_cast<float>(Ld + 1) - static_cast<float>(l) + 1.0f, 0.0f, 1.0f) : 1.0f;
                if (reveal <= 0.0f) continue;
                int in_st = l == 0 ? sg.in_stage : sg.layer_stages[static_cast<size_t>(l - 1)];
                int out_st = sg.layer_stages[static_cast<size_t>(l)];
                const DSceneNode& dst = n.nodes[static_cast<size_t>(n.node(out_st, q))];
                const DSceneNode& res = n.nodes[static_cast<size_t>(n.node(in_st, q))];
                cv.line_add(res.sx, res.sy, dst.sx, dst.sy, gold(), 0.18f * fade * reveal);
                for (int kk = 0; kk < Kself; ++kk) {
                    size_t at = (static_cast<size_t>(tt) * Ld + l) * Kself + kk;
                    int id = sg.tape.self_id.iv(at);
                    float w = sg.tape.self_w.f(at);
                    if (id < 0 || id >= Pn || w < 0.03f) continue;
                    const DSceneNode& src = n.nodes[static_cast<size_t>(n.node(in_st, id))];
                    cv.line_add(src.sx, src.sy, dst.sx, dst.sy, mix(violet(), gold(), std::min(1.0f, w * 1.8f)),
                                std::min(1.0f, (0.06f + 0.7f * std::sqrt(w)) * fade * reveal));
                }
                if (sg.cross_stage >= 0 && Kcross > 0) {
                    for (int kk = 0; kk < Kcross; ++kk) {
                        size_t at = (static_cast<size_t>(tt) * Ld + l) * Kcross + kk;
                        int id = sg.tape.cross_id.iv(at);
                        float w = sg.tape.cross_w.f(at);
                        if (id < 0 || id >= Ccross || w < 0.03f) continue;
                        const DSceneNode& src = n.nodes[static_cast<size_t>(n.node(sg.cross_stage, id))];
                        float al = (0.10f + 0.85f * std::sqrt(w)) * fade * reveal;
                        cv.line_add(src.sx, src.sy, dst.sx, dst.sy, mix(magenta(), gold(), std::min(1.0f, w * 1.6f)), std::min(1.0f, al));
                    }
                }
            }
            if (age == 0 && Ld > 0) {
                const DSceneNode& last = n.nodes[static_cast<size_t>(n.node(sg.layer_stages.back(), q))];
                int cand_cnt = std::min(demo.top_k, n.count[static_cast<size_t>(sg.out_stage)]);
                for (int o = 0; o < cand_cnt; ++o) {
                    int id = sg.tape.top_id.iv(static_cast<size_t>(tt) * demo.top_k + o);
                    if (id < 0) continue;
                    const DSceneNode& cand = n.nodes[static_cast<size_t>(n.node(sg.out_stage, o))];
                    float pr = sg.tape.top_p.f(static_cast<size_t>(tt) * demo.top_k + o);
                    cv.line_add(last.sx, last.sy, cand.sx, cand.sy, o == 0 ? gold() : violet(),
                                (0.08f + 0.9f * pr) * std::clamp(static_cast<float>(nw.frac) * static_cast<float>(Ld + 1) - static_cast<float>(Ld) + 1.0f, 0.0f, 1.0f));
                }
            }
        }
        // Flowing step pulses
        for (int l = 0; l < Ld; ++l) {
            const int q = std::clamp(t - 1, 0, n.count[static_cast<size_t>(sg.layer_stages[static_cast<size_t>(l)])] - 1);
            const DSceneNode& dst = n.nodes[static_cast<size_t>(n.node(sg.layer_stages[static_cast<size_t>(l)], q))];
            int in_st = l == 0 ? sg.in_stage : sg.layer_stages[static_cast<size_t>(l - 1)];
            for (int kk = 0; kk < std::min(3, Kself); ++kk) {
                size_t at = (static_cast<size_t>(t) * Ld + l) * Kself + kk;
                int id = sg.tape.self_id.iv(at);
                float w = sg.tape.self_w.f(at);
                if (id < 0 || id >= n.count[static_cast<size_t>(in_st)] || w < 0.14f) continue;
                const DSceneNode& src = n.nodes[static_cast<size_t>(n.node(in_st, id))];
                float ph = std::fmod(static_cast<float>(nw.frac) * 1.3f + hash01(static_cast<uint32_t>(id), static_cast<uint32_t>(l)) * 0.3f, 1.0f);
                cv.glow(src.sx + (dst.sx - src.sx) * ph, src.sy + (dst.sy - src.sy) * ph, 3.0f, rgb(0xfff2c0), std::min(1.0f, w * 2.0f));
            }
        }
    }

    const float nmax2 = nw.reading ? 1.0f : norm_max(nw.seg);
    const size_t n_chan = sg.tape.norm.dim(2);
    for (size_t i = 0; i < n.nodes.size(); ++i) {
        const DSceneNode& nd = n.nodes[i];
        if (nd.stage == sg.out_stage) continue;
        float g = 0.0f, al = 0.15f, r = 1.4f;
        float crowd = std::clamp(70.0f / static_cast<float>(std::max(1, n.count[static_cast<size_t>(nd.stage)])), 0.22f, 1.0f);
        Color col = rgb(0x9a90ff);
        if (nd.stage >= sg.in_stage) {
            int q = nd.idx;
            bool written = !nw.reading && q + 1 <= t;
            int layer = -1;
            for (int l = 0; l < Ld; ++l) if (sg.layer_stages[static_cast<size_t>(l)] == nd.stage) layer = l;
            float act_v = 0.2f;
            if (written && layer >= 0) act_v = sg.tape.norm.f((static_cast<size_t>(q + 1) * Ld + layer) * n_chan) / nmax2;
            g = written ? std::min(1.0f, act_v) : 0.0f;
            col = mix(rgb(0x6e66c8), gold(), g);
            al = written ? std::min(1.0f, (0.12f + 0.8f * g) * (std::sqrt(crowd) + 0.15f)) : 0.10f;
            r = (0.9f + 1.4f * g) * std::sqrt(crowd) + 0.3f;
        } else if (nd.stage == sg.cross_stage) {
            float uu = cross_use[static_cast<size_t>(nd.idx)] / cuse_max;
            g = nw.reading ? 0.25f : std::sqrt(uu);
            col = mix(rgb(0xb9a24a), gold(), g);
            al = std::min(1.0f, 0.35f + 0.65f * g);
            r = 2.8f + 2.2f * g;
        } else {
            float lit = nw.reading ? (0.35f + 1.3f * static_cast<float>(std::exp(-std::pow(static_cast<double>(nd.stage) - wave, 2.0) / 1.3))) : 1.0f;
            g = nd.act * 0.6f;
            col = mix(rgb(0x9a90ff), gold(), std::min(1.0f, g * 0.9f));
            al = std::min(1.0f, (0.10f + 0.4f * nd.act) * (std::sqrt(crowd) + 0.15f) * lit);
            r = (0.9f + 1.0f * nd.act) * std::sqrt(crowd) + 0.3f;
        }
        r *= std::min(1.8f, std::max(0.7f, nd.scale * 1.1f));
        cv.glow(nd.sx, nd.sy, r * 2.0f, col, al * 0.7f);
        cv.add_px(static_cast<int>(nd.sx), static_cast<int>(nd.sy), col, std::min(1.0f, al + 0.25f));
    }

    if (!nw.reading && Ld > 0 && t - 1 >= 0 && t - 1 < n.count[static_cast<size_t>(sg.layer_stages.back())]) {
        const DSceneNode& cur = n.nodes[static_cast<size_t>(n.node(sg.layer_stages.back(), t - 1))];
        float pulse = 0.7f + 0.3f * std::sin(static_cast<float>(nw.frac) * 6.2832f);
        cv.glow(cur.sx, cur.sy, 12.0f * std::max(0.8f, cur.scale) * pulse, rgb(0xffe08a), 0.95f);
        int written_id = sg.ftok.iv(static_cast<size_t>(t));
        int cand_cnt = std::min(demo.top_k, n.count[static_cast<size_t>(sg.out_stage)]);
        for (int o = 0; o < cand_cnt; ++o) {
            const DSceneNode& cand = n.nodes[static_cast<size_t>(n.node(sg.out_stage, o))];
            int id = sg.tape.top_id.iv(static_cast<size_t>(t) * demo.top_k + o);
            if (id < 0) continue;
            float pr = sg.tape.top_p.f(static_cast<size_t>(t) * demo.top_k + o);
            bool chosen = id == written_id;
            cv.glow(cand.sx, cand.sy, (4.0f + 9.0f * std::sqrt(pr)) * std::max(0.8f, cand.scale), chosen ? gold() : rgb(0x8f88e8), chosen ? 0.95f : 0.6f);
            cv.text(static_cast<int>(cand.sx) + s(10), static_cast<int>(cand.sy) - s(7), demo.label(id), chosen ? gold() : pal::dim, F_SMALL);
        }
    }

    StaggeredLabels stagger;
    for (const DStageGroup& g : n.groups) {
        float x0, y0, x1, y1, sc0, sc1;
        c.map(n.stage_x[static_cast<size_t>(g.a)], -162.0f, 0.0f, &x0, &y0, &sc0);
        c.map(n.stage_x[static_cast<size_t>(g.b)], -162.0f, 0.0f, &x1, &y1, &sc1);
        bool lit = nw.reading ? (wave + 0.5 >= g.a && wave - 0.5 <= g.b) : (g.a >= sg.in_stage);
        Color col = lit ? pal::select : pal::dim;
        cv.line(static_cast<int>(x0), static_cast<int>(y0), static_cast<int>(x1), static_cast<int>(y1), col);
        int tw = cv.text_width(g.label, F_SMALL), cx = static_cast<int>((x0 + x1) / 2);
        int row = stagger.place(cx, tw, s(8));
        cv.text(cx - tw / 2, static_cast<int>((y0 + y1) / 2) + s(4) + row * (cv.fonts.height(F_SMALL) + s(2)), g.label, col, F_SMALL);
    }
    cv.pop_clip();
    std::string caption = nw.reading
                              ? fmt("%s: encoding source context (%d%%)", sg.label.c_str(), static_cast<int>(nw.read_p * 100))
                              : fmt("%s: generating token %d of %d", sg.label.c_str(), t, sg.free_n() - 1);
    cv.text_fit(view.x + s(12), view.y + s(8), view.w - s(24), caption, nw.reading ? pal::select : pal::text, F_BOLD);
    cv.text_fit(view.x + s(12), view.y + s(28), view.w - s(24), "Drag to orbit 3D view, scroll wheel to zoom", pal::faint, F_TINY);
    add_hit(view, DH_SCENE);
}

void DemoApp::draw_text(Rect r) {
    cv.fill(r, pal::panel);
    cv.frame(r, pal::border);
    cv.push_clip(r);
    Now nw = now();
    const DemoSegment& sg = demo.segments[static_cast<size_t>(nw.seg)];
    const int m = sg.free_n(), n_ref = static_cast<int>(sg.ref_tok.n), t = nw.reading ? sg.t0 - 1 : nw.token;
    int x = r.x + s(12), y = r.y + s(10), w = r.w - s(24);
    cv.text_fit(x, y, w, sg.label, pal::text, F_BIG);
    y += s(28);
    cv.text_fit(x, y, w, n_ref > 0 ? fmt("generated %d tokens (reference %d)%s", m, n_ref, sg.free_exact ? " - exact" : "") : fmt("generated %d tokens", m),
                pal::dim, F_SMALL);
    y += s(22);

    Rect strip = {x, y, w, static_cast<int>(r.h * 0.40)};
    const int cw = m <= 24 ? s(40) : (m <= 48 ? s(32) : s(24)), ch = s(20), row_h = s(24);
    const int per_row = std::max(1, strip.w / cw), rows_visible = std::max(1, strip.h / row_h);
    const int cur_row = std::max(0, t) / per_row, first_row = std::max(0, cur_row - rows_visible + 3);
    const int show_to = ghost ? std::max(m, n_ref) : std::max(t + 1, sg.t0);
    cv.push_clip(strip);
    for (int i = 0; i < show_to; ++i) {
        int row = i / per_row - first_row;
        if (row < 0 || row >= rows_visible) continue;
        Rect cell = {strip.x + (i % per_row) * cw, strip.y + row * row_h, cw - 2, ch};
        bool written = i <= t && i < m;
        std::string lab;
        Color col = pal::panel2;
        if (written) {
            int wt = sg.ftok.iv(static_cast<size_t>(i));
            bool is_ctx = i < sg.t0;
            bool same = i < n_ref && sg.ref_tok.iv(static_cast<size_t>(i)) == wt;
            float p = sg.fp.f(static_cast<size_t>(i));
            col = is_ctx ? pal::panel2 : (n_ref == 0 || same ? mix(rgb(0x1d3a33), rgb(0x2f7d68), std::clamp(p, 0.0f, 1.0f)) : rgb(0x8a2a38));
            if (i == t && !nw.reading) col = mix(col, gold(), 0.55f);
            lab = demo.label(wt);
        } else if (i < n_ref) {
            col = rgb(0x1a1f26);
            lab = demo.label(sg.ref_tok.iv(static_cast<size_t>(i)));
        }
        cv.fill(cell, col);
        if (!written) cv.frame(cell, rgb(0x2b323c));
        FontId f = cv.text_width(lab, F_SMALL) <= cw - 3 ? F_SMALL : F_TINY;
        if (!lab.empty()) cv.text_fit(cell.x + 1, cell.y + (ch - cv.fonts.height(f)) / 2, cw - 2, lab, written ? text_on(col) : pal::faint, f, 1);
        if (i == t && !nw.reading) cv.frame(cell.inset(-1), pal::select, 2);
        if (i == sg.diverge) cv.frame(cell.inset(-1), pal::warn, 1);
    }
    cv.pop_clip();
    y += strip.h + s(10);

    if (nw.reading) {
        cv.text_fit(x, y, w, fmt("Encoding source tokens: %d%%", static_cast<int>(nw.read_p * 100)), pal::select, F_BOLD);
        y += s(24);
        Rect bar = {x, y, w, s(10)};
        cv.fill(bar, pal::panel2);
        cv.fill({bar.x, bar.y, static_cast<int>(bar.w * nw.read_p), bar.h}, pal::select);
        cv.pop_clip();
        return;
    }

    const size_t tt = static_cast<size_t>(t);
    int written_id = sg.ftok.iv(tt);
    cv.text(x, y, fmt("step %d", t), pal::dim, F_SMALL);
    cv.text(x + s(64), y - s(4), "\"" + demo.label(written_id) + "\"", pal::text, F_BIG);
    y += s(30);
    cv.text_fit(x, y, w, fmt("prob: %.3f   entropy: %.2f bits", sg.fp.f(tt), sg.tape.entropy.f(tt)), pal::text, F_SMALL);
    y += s(20);
    for (int o = 0; o < demo.top_k; ++o) {
        int id = sg.tape.top_id.iv(tt * demo.top_k + o);
        if (id < 0) break;
        float pr = sg.tape.top_p.f(tt * demo.top_k + o);
        bool chosen = id == written_id;
        cv.text_fit(x, y, s(70), demo.label(id), chosen ? gold() : pal::text, F_MONO, 2);
        Rect b = {x + s(76), y + s(3), w - s(142), s(11)};
        cv.fill(b, pal::panel2);
        cv.fill({b.x, b.y, static_cast<int>(b.w * std::clamp(pr, 0.0f, 1.0f)), b.h}, chosen ? gold() : pal::accent);
        cv.text(b.right() + s(6), y, fmt("%.3f", pr), pal::dim, F_SMALL);
        y += s(19);
    }
    cv.pop_clip();
}

void DemoApp::draw_sidebar(Rect r) {
    cv.fill(r, pal::panel);
    cv.frame(r, pal::border);
    cv.text(r.x + s(10), r.y + s(8), "Segments & Spans", pal::dim, F_SMALL);
    cv.push_clip(r);
    Now nw = now();
    int y = r.y + s(30);
    for (size_t i = 0; i < demo.segments.size(); ++i) {
        const DemoSegment& sg = demo.segments[i];
        Rect row = {r.x + s(6), y, r.w - s(12), s(24)};
        bool cur = static_cast<int>(i) == nw.seg;
        cv.fill(row, cur ? rgb(0x2a3f5e) : pal::panel2);
        if (cur) cv.frame(row, pal::accent);
        cv.fill({row.x + s(6), row.y + s(7), s(8), s(10)}, sg.ref_tok.empty() ? pal::accent : (sg.free_exact ? pal::good : pal::bad));
        cv.text_fit(row.x + s(20), row.y + s(4), row.w - s(80), sg.label, pal::text, F_SMALL);
        cv.text(row.x, row.y + s(4), fmt("%d tok", sg.free_n()), pal::dim, F_SMALL, row.w - s(8), 2);
        add_hit(row, DH_SIDE_SEG, static_cast<int>(i));
        y += s(28);
    }
    if (!demo.spans.empty()) {
        y += s(8);
        cv.text(r.x + s(10), y, "Annotated Spans", pal::dim, F_SMALL);
        y += s(20);
        for (const SpanInfo& sp : demo.spans) {
            Rect row = {r.x + s(8), y, r.w - s(16), s(22)};
            if (row.bottom() > r.bottom() - s(6)) break;
            Color tint = sp.tint.empty() ? pal::accent : parse_tint(sp.tint, pal::accent);
            cv.fill(row, pal::panel2);
            cv.fill({row.x, row.y, s(4), row.h}, tint);
            cv.text_fit(row.x + s(8), row.y + s(3), row.w - s(60), sp.label, pal::text, F_SMALL);
            cv.text(row.x, row.y + s(4), fmt("%d..%d", sp.start, sp.end - 1), pal::dim, F_TINY, row.w - s(6), 2);
            add_hit(row, DH_SPAN_JUMP, sp.start);
            y += s(25);
        }
    }
    cv.pop_clip();
}

void DemoApp::draw_transport(Rect r) {
    cv.fill(r, pal::panel);
    cv.hline(0, cv.W - 1, r.y, pal::border);
    Now nw = now();
    Rect tl = {r.x + s(16), r.y + s(14), r.w - s(32), s(28)};
    timeline_rect_ = tl;
    cv.fill(tl, rgb(0x14181c));
    double tot = std::max(1e-6, demo.total);
    for (const DemoSegment& sg : demo.segments) {
        int x0 = tl.x + static_cast<int>(sg.start / tot * tl.w), x1 = tl.x + static_cast<int>(sg.end() / tot * tl.w);
        int xr = tl.x + static_cast<int>((sg.start + sg.read_ticks) / tot * tl.w);
        Color base = sg.ref_tok.empty() ? rgb(0x2d587a) : (sg.free_exact ? rgb(0x2a6a4a) : rgb(0x7a3340));
        if (xr > x0) cv.fill({x0, tl.y + s(6), xr - x0, tl.h - s(12)}, mix(base, rgb(0x14181c), 0.45f));
        cv.fill({xr, tl.y + s(6), std::max(1, x1 - xr), tl.h - s(12)}, base);
        if (sg.diverge >= sg.t0 && sg.diverge < sg.free_n()) {
            int xd = tl.x + static_cast<int>((sg.start + sg.read_ticks + (sg.diverge - sg.t0)) / tot * tl.w);
            cv.fill({xd - 1, tl.y, 3, s(6)}, pal::warn);
        }
    }
    cv.frame(tl, pal::border);
    int cx = tl.x + static_cast<int>(std::clamp(pos, 0.0, tot) / tot * tl.w);
    cv.fill({cx - 1, tl.y - s(4), 3, tl.h + s(8)}, rgb(0xffffff));
    cv.fill({cx - s(5), tl.y - s(6), s(11), s(7)}, pal::select);
    add_hit(tl, DH_TIMELINE);

    int by = r.y + s(54), bh = s(30), x = r.x + s(16);
    const float fit = std::min(1.0f, static_cast<float>(r.w - s(32)) / (960.0f * cv.fonts.scale()));
    auto fs = [&](int u) { return std::max(1, static_cast<int>(static_cast<float>(s(u)) * fit + 0.5f)); };
    auto btn = [&](const std::string& t, bool active, int kind, int a, int wu) {
        button({x, by, fs(wu), bh}, t, active, kind, a, F_SMALL);
        x += fs(wu) + fs(6);
    };
    btn("Prev Seg", false, DH_PREV_SEG, 0, 82);
    btn("Step -", false, DH_STEP_BACK, 0, 68);
    btn(playing ? "Pause" : "Play", playing, DH_PLAY, 0, 72);
    btn("Step +", false, DH_STEP_FWD, 0, 68);
    btn("Next Seg", false, DH_NEXT_SEG, 0, 82);
    x += fs(14);
    static const char* names[N_SPEEDS] = {"0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x"};
    for (int i = 0; i < N_SPEEDS; ++i) btn(names[i], std::fabs(SPEEDS[i] - speed) < 1e-9, DH_SPEED, i, 44);
    x += fs(14);
    btn("Loop", loop, DH_LOOP, 0, 60);
    btn("Ref", ghost, DH_GHOST, 0, 52);

    const DemoSegment& sg = demo.segments[static_cast<size_t>(nw.seg)];
    std::string where = fmt("%s | %s", sg.label.c_str(), nw.reading ? "reading source" : fmt("token %d", nw.token).c_str());
    cv.text_fit(r.x + s(16), by + bh + s(10), r.w - s(32), status_.empty() ? where : where + " | " + status_, pal::dim, F_SMALL);
}

void DemoApp::draw_library(Rect r) {
    if (library_dirty_) scan_library();
    cv.fill(r, pal::bg);
    int x = r.x + s(40), y = r.y + s(30), w = std::min(r.w - s(80), s(1000));
    cv.text(x, y, "Demo Library", pal::text, F_BIG);
    y += s(34);
    cv.text_fit(x, y, w, "Select a .vtdemo recording, drag & drop a file into the window, or press Ctrl+O.", pal::dim, F_UI);
    y += s(28);
    if (!error_.empty()) { cv.text_fit(x, y, w, error_, pal::bad, F_UI); y += s(26); }
    button({x, y, s(180), s(32)}, "Open File...", false, DH_LIB_OPEN, 0, F_UI);
    y += s(52);
    if (library_items_.empty()) {
        cv.text_fit(x, y, w, "No .vtdemo files found yet. Record one with vitro.record_demo(...) in Python.", pal::dim, F_UI);
        lib_sv_ = ScrollView{};
        return;
    }
    const int row_h = s(52);
    Rect list = {r.x, y, r.w, r.bottom() - y};
    lib_sv_ = {list, static_cast<int>(library_items_.size()) * row_h};
    library_scroll_ = std::clamp(library_scroll_, 0, lib_sv_.max_offset());
    cv.push_clip(list);
    for (size_t i = 0; i < library_items_.size(); ++i) {
        const LibraryItem& it = library_items_[i];
        Rect row = {x, y - library_scroll_ + static_cast<int>(i) * row_h, w, s(46)};
        if (row.bottom() < list.y || row.y > list.bottom()) continue;
        bool hov = row.intersect(list).contains(mx_, my_);
        cv.fill(row, hov ? rgb(0x2a3f5e) : pal::panel);
        cv.frame(row, hov ? pal::accent : pal::border);
        cv.text_fit(row.x + s(14), row.y + s(5), row.w - s(240), it.name, pal::text, F_UI);
        cv.text_fit(row.x + s(14), row.y + s(25), row.w - s(240), it.path, pal::faint, F_SMALL);
        cv.text(row.x, row.y + s(6), fmt("%.2f MB", static_cast<double>(it.bytes) / (1 << 20)), pal::dim, F_SMALL, row.w - s(14), 2);
        add_hit(row.intersect(list), DH_LIB_ITEM, static_cast<int>(i));
    }
    cv.pop_clip();
    Rect track = draw_scrollbar(cv, lib_sv_, library_scroll_, scrolling_);
    if (!track.empty()) add_hit(track, DH_SCROLL, 0);
}

void DemoApp::draw_header() {
    Rect hd = {0, 0, cv.W, s(54)};
    cv.fill(hd, pal::panel);
    cv.hline(0, cv.W - 1, hd.bottom() - 1, pal::border);
    cv.text(s(14), s(8), "Vitro Demo", pal::text, F_BIG);
    int w1 = cv.text_width("Vitro Demo", F_BIG);
    if (demo.loaded()) {
        cv.text(s(14) + w1 + s(14), s(13), demo.title, pal::accent, F_MONO);
        cv.text_fit(s(14), s(34), cv.W - s(28), fmt("%s (%s) | %d segments, %d tokens",
                                                  demo.model_name.c_str(), demo.arch.c_str(), static_cast<int>(demo.segments.size()), demo.total_tokens),
                    pal::dim, F_SMALL);
    } else {
        cv.text_fit(s(14), s(34), cv.W - s(28), "Interactive 3D replay of neural network autoregressive generation.", pal::dim, F_SMALL);
    }
}

void DemoApp::draw_help() {
    Rect r = {cv.W / 2 - s(330), cv.H / 2 - s(230), s(660), s(460)};
    cv.fill(r, rgb(0x14181c), 0.97f);
    cv.frame(r, pal::accent);
    cv.text(r.x + s(16), r.y + s(12), "Vitro Demo Controls", pal::text, F_BIG);
    static const char* lines[] = {
        "Space            Play / pause",
        "Left / Right     Step backward / forward (Shift = 10)",
        "Up / Down        Increase / decrease playback speed",
        "[ / ]            Previous / next segment",
        "Home / End       Jump to start / end",
        "D / L / G / R    Divergence / Loop / Reference overlay / Reset 3D camera",
        "B / Ctrl+O       Demo library / Open .vtdemo file",
        "F11 / S / Q      Fullscreen / Screenshot / Quit",
    };
    int y = r.y + s(54);
    for (const char* ln : lines) { cv.text(r.x + s(20), y, ln, pal::text, F_MONO); y += s(26); }
}

void DemoApp::describe(const DemoHit& h) {
    switch (h.kind) {
        case DH_PLAY: status_ = "Play / pause (Space)"; break;
        case DH_STEP_BACK: status_ = "Step backward (Left)"; break;
        case DH_STEP_FWD: status_ = "Step forward (Right)"; break;
        case DH_PREV_SEG: status_ = "Previous segment ([)"; break;
        case DH_NEXT_SEG: status_ = "Next segment (])"; break;
        case DH_SPEED: status_ = fmt("Speed %.2gx", SPEEDS[std::clamp(h.a, 0, N_SPEEDS - 1)]); break;
        case DH_LOOP: status_ = "Toggle looping (L)"; break;
        case DH_GHOST: status_ = "Toggle reference token ghost cells (G)"; break;
        case DH_TIMELINE: status_ = "Click or drag to scrub timeline"; break;
        case DH_SIDE_SEG: status_ = "Jump to segment"; break;
        case DH_SPAN_JUMP: status_ = "Jump to annotated span"; break;
        default: break;
    }
}

void DemoApp::render() {
    hits_.clear();
    cv.clear(pal::bg);
    const int hh = s(54), th = s(116);
    if (!demo.loaded() || library_) {
        draw_header();
        draw_library({0, hh, cv.W, cv.H - hh});
        status_.clear();
        if (const DemoHit* h = hit_at(mx_, my_)) describe(*h);
        if (notice_left_ > 0) status_ = notice_text_;
        if (!status_.empty()) cv.text_fit(s(14), cv.H - s(26), cv.W - s(28), status_, pal::dim, F_SMALL);
        if (help_) draw_help();
        return;
    }
    Rect side = {0, hh, s(250), cv.H - hh - th};
    int text_w = std::min(s(470), cv.W * 36 / 100);
    Rect text = {cv.W - text_w, hh, text_w, cv.H - hh - th};
    Rect scene = {side.right(), hh, text.x - side.right(), cv.H - hh - th};
    status_.clear();
    draw_scene(scene);
    draw_sidebar(side);
    draw_text(text);
    if (const DemoHit* h = hit_at(mx_, my_)) describe(*h);
    if (notice_left_ > 0) status_ = notice_text_;
    draw_transport({0, cv.H - th, cv.W, th});
    draw_header();
    if (help_) draw_help();
}

}  // namespace vitro
