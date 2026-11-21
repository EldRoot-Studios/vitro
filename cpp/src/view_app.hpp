// Vitro trace viewer application header: UI state, modular 5-view dispatch, helpers.
#pragma once

#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vitro/canvas.hpp>
#include <vitro/colors.hpp>
#include <vitro/math3d.hpp>
#include <vitro/scrollbar.hpp>
#include <vitro/shell.hpp>
#include "vitro_file.hpp"

namespace vitro {

inline constexpr float NET_YAW0 = 0.30f, NET_PITCH0 = 0.14f;

enum class View { Tokens = 0, Attention, Activations, Spans, Net };
inline constexpr int VIEW_COUNT = 5;
const char* view_name(View v);

enum HitKind {
    H_NONE = 0, H_TAB, H_SIDE_UNIT, H_TOKEN, H_FREE_TOKEN, H_SPAN_ROW, H_ATTN_FAM, H_LAYER, H_HEAD, H_GRID,
    H_COLOR, H_OVERLAY, H_MAP_ATTN, H_GRID_CELL, H_NORM_LAYER, H_LATENT_SEL, H_LATENT_CELL, H_LATENT_COS,
    H_NET_VIEW, H_NET_BTN, H_LIB_ITEM, H_LIB_OPEN, H_SCROLL
};

struct Hit {
    Rect r;
    int kind = H_NONE;
    int a = 0, b = 0;
};

struct ViewState {
    View view = View::Tokens;
    int unit = 0;               // selected UnitTrace index
    int token = -1;             // selected target sequence token index
    int attn_fam = 0;           // selected AttnFamily in active unit
    int layer = -1;             // -1 = mean across layers
    int head = -1;              // -1 = mean across heads
    bool grid = false;
    int latent_idx = 0;         // selected LatentMap index in active unit
    int latent_row = -1;
    int span_sel = -1;
    int color_mode = 0;         // 0 surprisal (+ numeric delta if present), 1 surprisal, 2 rank, 3 entropy
    bool overlay = true;
    int side_scroll = 0;
    int scroll = 0;
    bool help = false;
    int per_row = 40;
    int cell_w = 0;
    // 3D network view
    float net_yaw = NET_YAW0, net_pitch = NET_PITCH0, net_zoom = 1.0f;
    double net_time = 0.0;
    bool net_play = true;
    bool net_path = false;
    int net_k = 3;
};

class VitroApp final : public ShellApp {
public:
    TraceDump d;
    ViewState st;
    Canvas cv;
    std::vector<Hit> hits;
    int mx = -1, my = -1;
    bool dragging = false;
    int press_x = 0, press_y = 0;
    bool moved = false;
    std::shared_ptr<void> net;
    std::string status;
    std::string recent_file, library_dir;
    ScrollView main_sv, side_sv;
    std::vector<std::string> recent;
    std::vector<LibraryItem> library_items;
    bool library_dirty = true, library = false;
    int library_scroll = 0;
    std::string error, notice_text;
    double notice_left = 0.0;
    bool adaptive = false;

    VitroApp(int w, int h, float base_scale) : cv(w, h, base_scale), base_scale_(base_scale) {}

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
    void notice(const std::string& text) override { notice_text = text; notice_left = 6.0; }
    std::string open_file(const std::string& path) override;
    std::string title() override;
    float ui_zoom() const override { return zoom_; }
    void set_ui_zoom(float zoom) override;
    std::vector<Menu> menus() override;
    void command(int id) override;
    std::string state_json() override;
    std::string status_line() const override { return status; }

    void click(int x, int y);
    void load_recent();
    void remember(std::string path);
    void scan_library();
    bool start_screen() const { return !d.loaded() || library; }

    int header_h() const { return cv.s(58); }
    int side_w() const { return cv.s(262); }
    int status_h() const { return cv.s(24); }
    Rect main_rect() const { return {side_w(), header_h(), cv.W - side_w(), cv.H - header_h() - status_h()}; }

    const UnitTrace& U() const { return d.units[static_cast<size_t>(st.unit)]; }
    void select_unit(int u);
    void select_token(int j);
    void clamp();

    void add_hit(Rect r, int kind, int a = 0, int b = 0) { hits.push_back({r, kind, a, b}); }
    const Hit* hit_at(int x, int y) const;

    bool button(Rect r, const std::string& label, bool active, int kind, int a = 0, int b = 0, FontId f = F_SMALL);
    void panel(Rect r, const std::string& title);
    void bar(Rect r, double frac, Color c);

    std::string token_text(const UnitTrace& u, int j) const { return j >= 0 && j < u.n_tokens() ? d.label(u.tok.iv(static_cast<size_t>(j))) : "?"; }
    std::string source_text(const UnitTrace& u, int j) const {
        if (u.src_tok.empty()) return token_text(u, j);
        return j >= 0 && j < u.n_source() ? d.label(u.src_tok.iv(static_cast<size_t>(j))) : "?";
    }
    std::string free_text(const UnitTrace& u, int j) const { return j >= 0 && j < u.free_n() ? d.label(u.ftok.iv(static_cast<size_t>(j))) : std::string(); }
    bool is_scalar_token(const UnitTrace& u, int j) const;
    Color cell_color(const UnitTrace& u, int j, bool* wrong) const;
    std::string describe_token(const UnitTrace& u, int j) const;
    std::vector<const SpanInfo*> spans_covering(int j) const;

    // Averaged/indexed attention matrix for family `fam` of unit `u` (rows x cols). Cached across frames.
    const std::vector<float>& attn_map(int u_idx, int fam_idx, int layer, int head);
    // Cosine similarity between rows of LatentMap `lat_idx` (S x S).
    const std::vector<float>& latent_cos(int u_idx, int lat_idx);

private:
    std::map<std::array<int, 4>, std::vector<float>> attn_cache_;
    std::map<std::array<int, 2>, std::vector<float>> cos_cache_;
    float base_scale_ = 1.0f, zoom_ = 1.0f;
    int scroll_drag_ = -1;
    int request_ = 0;

    int& scroll_ref(int which);
    ScrollView& scroll_view(int which) { return which == 1 ? side_sv : main_sv; }
    bool scroll_key(int vk);
    void draw_bars();
};

std::string fmt(const char* f, ...);
void cell_of(const VitroApp& a, const Hit& h, int* row, int* col);
void draw_tokens(VitroApp& a, Rect area);
void draw_attention(VitroApp& a, Rect area);
void draw_activations(VitroApp& a, Rect area);
void draw_spans(VitroApp& a, Rect area);
void draw_net(VitroApp& a, Rect area);
void draw_library(VitroApp& a, Rect area);
void draw_start_header(VitroApp& a);
void draw_start_status(VitroApp& a);
void describe_hit(VitroApp& a, const Hit& h);
void describe_library(VitroApp& a, const Hit& h);
bool describe_attention(VitroApp& a, const Hit& h);
bool describe_activations(VitroApp& a, const Hit& h);
bool describe_spans(VitroApp& a, const Hit& h);
bool describe_net(VitroApp& a, const Hit& h);
void net_pick_click(VitroApp& a, int x, int y);
void net_drag(VitroApp& a, int dx, int dy);
int run_selftest();

}  // namespace vitro
