// Vitro 3D generation player header: dynamic prefill + token-by-token neural step visualization.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <vitro/canvas.hpp>
#include <vitro/colors.hpp>
#include <vitro/math3d.hpp>
#include <vitro/scrollbar.hpp>
#include <vitro/shell.hpp>
#include "demo_file.hpp"

namespace vitro {

std::string fmt(const char* f, ...);

inline constexpr double TICKS_PER_SECOND = 10.0;

enum DemoHitKind {
    DH_NONE = 0, DH_PLAY, DH_STEP_BACK, DH_STEP_FWD, DH_PREV_SEG, DH_NEXT_SEG, DH_SPEED, DH_LOOP, DH_GHOST, DH_TIMELINE, DH_SCENE,
    DH_SIDE_SEG, DH_SPAN_JUMP, DH_LIB_ITEM, DH_LIB_OPEN, DH_SCROLL
};

struct DemoHit {
    Rect r;
    int kind = DH_NONE;
    int a = 0;
};

struct DSceneNode {
    float x = 0, y = 0, z = 0;
    int stage = 0, idx = 0;
    float act = 0;
    float sx = 0, sy = 0, scale = 1;
};

struct DSceneEdge {
    int a = 0, b = 0;
    float w = 0;
    int stage = 0;
    Color tint = rgb(0x7b6df0);
};

struct DStageGroup {
    int a = 0, b = 0;
    std::string label;
};

struct DemoScene {
    int seg = -1;
    std::vector<DSceneNode> nodes;
    std::vector<DSceneEdge> edges;
    std::vector<int> first, count;
    std::vector<float> stage_x;
    std::vector<DStageGroup> groups;
    float half_span = 100.0f;
    int node(int stage, int i) const { return first[static_cast<size_t>(stage)] + i; }
};

class DemoApp final : public ShellApp {
public:
    DemoApp(int w, int h, float base_scale) : cv(w, h, base_scale), base_scale_(base_scale) {}

    Canvas& canvas() override { return cv; }
    void render() override;
    void move(int x, int y) override;
    void press(int x, int y) override;
    void release(int x, int y) override;
    void wheel(int x, int y, int delta, bool ctrl) override;
    bool key(int vk, bool ctrl, bool shift) override;
    bool tick(double seconds) override;
    bool escape() override;
    int take_request() override { int r = request_; request_ = 0; return r; }
    void notice(const std::string& text) override { notice_text_ = text; notice_left_ = 6.0; }
    std::string open_file(const std::string& path) override;
    std::string title() override;
    float ui_zoom() const override { return zoom_; }
    void set_ui_zoom(float zoom) override;
    std::vector<Menu> menus() override;
    void command(int id) override;
    std::string state_json() override;
    std::string status_line() const override { return status_; }

    Canvas cv;
    TraceDemo demo;
    double pos = 0.0;
    bool playing = true, loop = true, ghost = true;
    double speed = 1.0;
    float yaw = 0.42f, pitch = 0.16f, cam_zoom = 1.0f;
    std::string recent_file, library_dir;

    struct Now {
        int seg = 0;
        bool reading = false;
        double read_p = 0.0;
        int token = 1;
        double frac = 0.5;
    };
    Now now() const;
    void seek_segment(int seg);
    void seek_token(int seg, int token);
    void load_recent();

private:
    float base_scale_ = 1.0f, zoom_ = 1.0f;
    int mx_ = -1, my_ = -1;
    bool dragging_ = false, moved_ = false, scrub_ = false, rotating_ = false, scrolling_ = false;
    int press_x_ = 0, press_y_ = 0, request_ = 0, library_scroll_ = 0;
    double sway_ = 0.0, notice_left_ = 0.0;
    bool library_ = false, library_dirty_ = true, help_ = false;
    std::string status_, error_, notice_text_;
    std::vector<std::string> recent_;
    std::vector<LibraryItem> library_items_;
    ScrollView lib_sv_;
    Rect scene_rect_, timeline_rect_;
    std::vector<DemoHit> hits_;
    std::shared_ptr<DemoScene> scene_;
    std::vector<float> seg_norm_max_;

    int s(int v) const { return cv.s(v); }
    void add_hit(Rect r, int kind, int a = 0) { hits_.push_back({r, kind, a}); }
    const DemoHit* hit_at(int x, int y) const;
    bool button(Rect r, const std::string& label, bool active, int kind, int a = 0, FontId f = F_SMALL);
    void toggle_play();
    void step(int by);
    void jump_segment(int dir);
    void scrub_to(int x);
    void remember(std::string path);
    void scan_library();
    void click(int x, int y);
    float norm_max(int seg);
    std::shared_ptr<DemoScene> scene_for(int seg);
    void draw_scene(Rect view);
    void draw_text(Rect r);
    void draw_sidebar(Rect r);
    void draw_transport(Rect r);
    void draw_library(Rect r);
    void draw_header();
    void draw_help();
    void describe(const DemoHit& h);
};

}  // namespace vitro
