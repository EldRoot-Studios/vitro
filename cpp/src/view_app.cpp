// Vitro viewer application frame: header tabs, sidebar units, status bar, selection, caching, and input dispatch.
#include "view_app.hpp"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <stdexcept>

namespace vitro {

std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

const char* view_name(View v) {
    static const char* names[VIEW_COUNT] = {"tokens", "attention", "activations", "spans", "net"};
    return names[static_cast<int>(v)];
}

void cell_of(const VitroApp& a, const Hit& h, int* row, int* col) {
    *row = std::clamp(static_cast<int>(static_cast<long long>(a.my - h.r.y) * h.a / std::max(1, h.r.h)), 0, std::max(0, h.a - 1));
    *col = std::clamp(static_cast<int>(static_cast<long long>(a.mx - h.r.x) * h.b / std::max(1, h.r.w)), 0, std::max(0, h.b - 1));
}

void VitroApp::select_unit(int u) {
    st.unit = u;
    st.token = -1;
    st.attn_fam = 0;
    st.layer = -1;
    st.head = -1;
    st.latent_idx = 0;
    st.latent_row = -1;
    st.scroll = 0;
    clamp();
}

void VitroApp::select_token(int j) {
    st.token = j;
    clamp();
}

void VitroApp::clamp() {
    if (!d.loaded()) return;
    st.unit = std::clamp(st.unit, 0, static_cast<int>(d.units.size()) - 1);
    const UnitTrace& u = U();
    st.token = std::clamp(st.token, -1, u.n_tokens() - 1);
    if (st.token >= 0 && st.token < u.t0 && u.n_tokens() > u.t0) st.token = u.t0;
    if (u.attn_families.empty()) {
        st.attn_fam = 0;
        st.layer = -1;
        st.head = -1;
    } else {
        st.attn_fam = std::clamp(st.attn_fam, 0, static_cast<int>(u.attn_families.size()) - 1);
        const AttnFamily& af = u.attn_families[static_cast<size_t>(st.attn_fam)];
        st.layer = std::clamp(st.layer, -1, af.layers - 1);
        st.head = std::clamp(st.head, -1, af.heads - 1);
    }
    if (u.latents.empty()) {
        st.latent_idx = 0;
        st.latent_row = -1;
    } else {
        st.latent_idx = std::clamp(st.latent_idx, 0, static_cast<int>(u.latents.size()) - 1);
        int rows = static_cast<int>(u.latents[static_cast<size_t>(st.latent_idx)].data.dim(0));
        st.latent_row = std::clamp(st.latent_row, -1, rows - 1);
    }
    st.span_sel = d.spans.empty() ? -1 : std::clamp(st.span_sel, -1, static_cast<int>(d.spans.size()) - 1);
    st.scroll = std::max(0, st.scroll);
    st.side_scroll = std::max(0, st.side_scroll);
}

const Hit* VitroApp::hit_at(int x, int y) const {
    for (size_t i = hits.size(); i-- > 0;)
        if (hits[i].r.contains(x, y)) return &hits[i];
    return nullptr;
}

bool VitroApp::is_scalar_token(const UnitTrace& u, int j) const {
    if (j < u.t0 || j >= u.n_tokens() || u.val_mean.empty()) return false;
    return !std::isnan(u.val_mean.f(static_cast<size_t>(j))) && d.has_value(u.tok.iv(static_cast<size_t>(j)));
}

Color VitroApp::cell_color(const UnitTrace& u, int j, bool* wrong) const {
    *wrong = false;
    if (j < u.t0) return pal::panel2;
    size_t jj = static_cast<size_t>(j);
    float p = u.p.f(jj);
    int rank = u.rank.iv(jj);
    *wrong = rank > 1;
    double bits = -std::log2(std::max(p, 1e-30f));
    auto by_rank = [&]() -> Color {
        if (rank <= 1) return rgb(0x26313f);
        if (rank == 2) return rgb(0xe6c84a);
        if (rank <= 5) return rgb(0xeb6e32);
        return rgb(0xdc2840);
    };
    switch (st.color_mode) {
        case 1: return surprisal(bits);
        case 2: return by_rank();
        case 3: return surprisal(u.entropy.f(jj));
        default: {
            if (is_scalar_token(u, j)) {
                int top = u.top_id.iv(jj * static_cast<size_t>(d.top_k));
                double v_true = d.value_of(u.tok.iv(jj));
                double v_top = d.has_value(top) ? d.value_of(top) : u.val_mean.f(jj);
                return delta_error(std::fabs(v_top - v_true), d.delta_red_at);
            }
            return surprisal(bits);
        }
    }
}

std::vector<const SpanInfo*> VitroApp::spans_covering(int j) const {
    std::vector<const SpanInfo*> out;
    for (const SpanInfo& s : d.spans)
        if (j >= s.start && j < s.end) out.push_back(&s);
    return out;
}

std::string VitroApp::describe_token(const UnitTrace& u, int j) const {
    if (j < 0 || j >= u.n_tokens()) return "";
    size_t jj = static_cast<size_t>(j);
    std::string s = fmt("token %d \"%s\"", j, token_text(u, j).c_str());
    if (!u.token_tags.empty()) {
        const std::string& tg = d.tag_name(u.token_tags.iv(jj));
        if (!tg.empty()) s += fmt(" [%s]", tg.c_str());
    }
    auto cov = spans_covering(j);
    if (!cov.empty()) s += fmt(" (%s)", cov.back()->label.c_str());
    if (j < u.t0) return s + " - context prefix";
    s += fmt(" - p=%.3f, rank=%d", u.p.f(jj), u.rank.iv(jj));
    int top = u.top_id.iv(jj * static_cast<size_t>(d.top_k));
    if (top >= 0 && u.rank.iv(jj) > 1) s += fmt(" (top-1: \"%s\" %.2f)", d.label(top).c_str(), u.top_p.f(jj * static_cast<size_t>(d.top_k)));
    if (is_scalar_token(u, j)) {
        double vt = d.value_of(u.tok.iv(jj));
        double vp = d.has_value(top) ? d.value_of(top) : u.val_mean.f(jj);
        s += fmt(", delta %.2g (target %.4g, top %.4g)", std::fabs(vp - vt), vt, vp);
    }
    return s;
}

const std::vector<float>& VitroApp::attn_map(int u_idx, int fam_idx, int layer, int head) {
    std::array<int, 4> key = {u_idx, fam_idx, layer, head};
    auto it = attn_cache_.find(key);
    if (it != attn_cache_.end()) return it->second;
    const UnitTrace& u = d.units[static_cast<size_t>(u_idx)];
    const AttnFamily& af = u.attn_families[static_cast<size_t>(fam_idx)];
    const size_t n = static_cast<size_t>(af.rows) * static_cast<size_t>(af.cols);
    std::vector<float> m(n, 0.0f);
    int l0 = layer < 0 ? 0 : layer, l1 = layer < 0 ? af.layers : layer + 1;
    int h0 = head < 0 ? 0 : head, h1 = head < 0 ? af.heads : head + 1;
    const float inv = 1.0f / static_cast<float>((l1 - l0) * (h1 - h0));
    for (int l = l0; l < l1; ++l) {
        const Arr& w = u.links[static_cast<size_t>(af.link_indices[static_cast<size_t>(l)])].weights;
        for (int h = h0; h < h1; ++h) {
            const size_t base = static_cast<size_t>(h) * n;
            for (size_t i = 0; i < n; ++i) m[i] += w.f(base + i) * inv;
        }
    }
    return attn_cache_.emplace(key, std::move(m)).first->second;
}

const std::vector<float>& VitroApp::latent_cos(int u_idx, int lat_idx) {
    std::array<int, 2> key = {u_idx, lat_idx};
    auto it = cos_cache_.find(key);
    if (it != cos_cache_.end()) return it->second;
    const LatentMap& lm = d.units[static_cast<size_t>(u_idx)].latents[static_cast<size_t>(lat_idx)];
    const size_t S = lm.data.dim(0), D = lm.data.dim(1);
    std::vector<float> norm(S * D, 0.0f);
    for (size_t s = 0; s < S; ++s) {
        double ss = 0.0;
        for (size_t c = 0; c < D; ++c) { double x = lm.data.f(s * D + c); ss += x * x; }
        float inv = ss > 1e-18 ? static_cast<float>(1.0 / std::sqrt(ss)) : 0.0f;
        for (size_t c = 0; c < D; ++c) norm[s * D + c] = lm.data.f(s * D + c) * inv;
    }
    std::vector<float> cos(S * S, 0.0f);
    for (size_t i = 0; i < S; ++i)
        for (size_t j = i; j < S; ++j) {
            double s = 0.0;
            for (size_t c = 0; c < D; ++c) s += norm[i * D + c] * norm[j * D + c];
            cos[i * S + j] = cos[j * S + i] = static_cast<float>(s);
        }
    return cos_cache_.emplace(key, std::move(cos)).first->second;
}

bool VitroApp::button(Rect r, const std::string& label, bool active, int kind, int a, int b, FontId f) {
    bool hov = r.contains(mx, my);
    cv.fill(r, active ? rgb(0x2f5f9e) : (hov ? rgb(0x39404a) : pal::panel2));
    cv.frame(r, active ? pal::accent : pal::border);
    cv.text_fit(r.x + cv.s(4), r.y + (r.h - cv.fonts.height(f)) / 2, std::max(1, r.w - cv.s(8)), label, active ? rgb(0xffffff) : pal::text, f, 1);
    add_hit(r, kind, a, b);
    return hov;
}

void VitroApp::panel(Rect r, const std::string& title) {
    cv.fill(r, pal::panel);
    cv.frame(r, pal::border);
    if (!title.empty()) cv.text(r.x + cv.s(10), r.y + cv.s(7), title, pal::dim, F_SMALL);
}

void VitroApp::bar(Rect r, double frac, Color c) {
    cv.fill(r, pal::panel2);
    int fill_w = static_cast<int>(std::clamp(frac, 0.0, 1.0) * r.w + 0.5);
    if (fill_w > 0) cv.fill({r.x, r.y, fill_w, r.h}, c);
}

int& VitroApp::scroll_ref(int which) {
    if (which == 1) return st.side_scroll;
    return start_screen() ? library_scroll : st.scroll;
}

void VitroApp::draw_bars() {
    for (int which = 0; which <= (start_screen() ? 0 : 1); ++which) {
        const ScrollView& sv = scroll_view(which);
        bool hot = scroll_drag_ == which || (sv.needed() && scrollbar_track(cv, sv).contains(mx, my));
        Rect tr = draw_scrollbar(cv, sv, scroll_ref(which), hot);
        if (!tr.empty()) add_hit(tr, H_SCROLL, which, 0);
    }
}

namespace {

void draw_header(VitroApp& a) {
    Canvas& cv = a.cv;
    const TraceDump& d = a.d;
    Rect hd = {0, 0, cv.W, a.header_h()};
    cv.fill(hd, pal::panel);
    cv.hline(0, cv.W - 1, hd.bottom() - 1, pal::border);
    cv.text(cv.s(12), cv.s(6), "Vitro", pal::text, F_BIG);
    const int x0 = cv.s(12) + cv.text_width("Vitro", F_BIG) + cv.s(14);
    const int tabs_w = cv.s(540);
    cv.text_fit(x0, cv.s(8), std::max(cv.s(80), cv.W - x0 - tabs_w - cv.s(20)), d.title, pal::accent, F_MONO);
    std::string sub = fmt("%s (%s)  |  %s / %s", d.model_name.c_str(), d.arch.c_str(), d.device.c_str(), d.precision.c_str());
    if (d.step >= 0) sub += fmt("  |  step %lld", d.step);
    if (!d.spans.empty()) sub += fmt("  |  %d spans", static_cast<int>(d.spans.size()));
    cv.text_fit(cv.s(12), cv.s(36), cv.W - tabs_w - cv.s(30), sub, pal::dim, F_SMALL);
    static const char* names[VIEW_COUNT] = {"1 Tokens", "2 Attention", "3 Activations", "4 Spans", "5 3D Net"};
    const int tw = cv.s(98), gap = cv.s(6), tx = cv.W - cv.s(12) - VIEW_COUNT * tw - (VIEW_COUNT - 1) * gap;
    for (int k = 0; k < VIEW_COUNT; ++k)
        a.button({tx + k * (tw + gap), cv.s(13), tw, cv.s(30)}, names[k], static_cast<int>(a.st.view) == k, H_TAB, k);
}

void draw_sidebar(VitroApp& a) {
    Canvas& cv = a.cv;
    const TraceDump& d = a.d;
    Rect sb = {0, a.header_h(), a.side_w(), cv.H - a.header_h() - a.status_h()};
    a.panel(sb, "Sequence Units");
    int y = sb.y + cv.s(30) - a.st.side_scroll, row_h = cv.s(28);
    Rect list_area = {sb.x, sb.y + cv.s(28), sb.w, sb.h - cv.s(82)};
    a.side_sv = {list_area, static_cast<int>(d.units.size()) * row_h + cv.s(8)};
    a.st.side_scroll = std::clamp(a.st.side_scroll, 0, a.side_sv.max_offset());
    cv.push_clip(list_area);
    for (size_t i = 0; i < d.units.size(); ++i) {
        const UnitTrace& u = d.units[i];
        Rect r = {sb.x + cv.s(6), y, sb.w - cv.s(18), cv.s(25)};
        y += row_h;
        if (r.bottom() < list_area.y || r.y > list_area.bottom()) continue;
        bool sel = static_cast<int>(i) == a.st.unit;
        bool hov = r.contains(a.mx, a.my);
        cv.fill(r, sel ? rgb(0x294466) : (hov ? pal::panel2 : pal::panel));
        if (sel) cv.frame(r, pal::accent);
        cv.fill({r.x + cv.s(5), r.y + cv.s(7), cv.s(9), cv.s(10)}, u.forced_exact ? pal::good : pal::bad);
        if (!u.ftok.empty()) cv.fill({r.x + cv.s(17), r.y + cv.s(7), cv.s(9), cv.s(10)}, u.free_exact ? pal::good : pal::bad);
        cv.text_fit(r.x + cv.s(32), r.y + cv.s(4), r.w - cv.s(92), u.label, sel ? rgb(0xffffff) : pal::text, F_SMALL);
        cv.text(r.x, r.y + cv.s(4), fmt("%d tok", u.n_tokens()), pal::dim, F_SMALL, r.w - cv.s(8), 2);
        a.add_hit(r.intersect(list_area), H_SIDE_UNIT, static_cast<int>(i));
    }
    cv.pop_clip();
    Rect lg = {sb.x + cv.s(10), sb.bottom() - cv.s(46), sb.w - cv.s(20), cv.s(38)};
    cv.fill({lg.x, lg.y + cv.s(4), cv.s(9), cv.s(10)}, pal::good);
    cv.text(lg.x + cv.s(14), lg.y, "forced top-1 exact", pal::dim, F_SMALL);
    cv.fill({lg.x, lg.y + cv.s(22), cv.s(9), cv.s(10)}, pal::good);
    cv.text(lg.x + cv.s(14), lg.y + cv.s(18), "free generation exact (box 2)", pal::dim, F_SMALL);
}

void draw_status(VitroApp& a) {
    Canvas& cv = a.cv;
    const TraceDump& d = a.d;
    Rect sr = {0, cv.H - a.status_h(), cv.W, a.status_h()};
    cv.fill(sr, pal::panel);
    cv.hline(0, cv.W - 1, sr.y, pal::border);
    std::string msg = a.notice_left > 0 ? a.notice_text : a.status;
    if (msg.empty()) msg = "Hover tokens or matrix cells; 1-5 switch views; F1 help";
    cv.text_fit(cv.s(10), sr.y + cv.s(4), cv.W - cv.s(260), msg, a.notice_left > 0 ? pal::good : pal::text, F_SMALL);
    int total_tok = 0, total_attn = 0;
    for (const UnitTrace& u : d.units) {
        total_tok += u.n_tokens();
        total_attn = std::max(total_attn, static_cast<int>(u.attn_families.size()));
    }
    cv.text(0, sr.y + cv.s(4), fmt("%d tokens | %d units | %d attn groups", total_tok, static_cast<int>(d.units.size()), total_attn), pal::dim, F_SMALL, cv.W - cv.s(10), 2);
}

void draw_help(VitroApp& a) {
    Canvas& cv = a.cv;
    cv.fill({0, 0, cv.W, cv.H}, rgb(0x0b0e14), 0.72f);
    Rect box = {cv.W / 2 - cv.s(360), cv.H / 2 - cv.s(250), cv.s(720), cv.s(500)};
    a.panel(box, "Keyboard & Mouse Controls");
    static const char* rows[] = {
        "1 - 5            Switch view: Tokens, Attention, Activations, Spans, 3D Net",
        "Left / Right     Previous / next token",
        "Up / Down        Next / previous strip row (or scroll active view)",
        "[ / ]            Previous / next sequence unit",
        "E / Shift+E      Jump to next / previous prediction error (rank > 1)",
        "D                Jump to first free-generation divergence",
        "L / Shift+L      Next / previous layer (-1 = layer average)",
        "H / Shift+H      Next / previous attention head (-1 = head average)",
        "F                Switch attention family (Self, Cross, etc.)",
        "G                Toggle all-layers x all-heads grid in Attention view",
        "C                Cycle token color mode (surprisal / rank / entropy / scalar delta)",
        "+ / - / 0        Resize token cells (0 resets auto-fit)",
        "3D Net:          Drag rotates, Wheel zooms, Space toggles wave, P rollout path, K edges, R reset",
        "Ctrl+O / Drag    Open .vtdump file",
        "B / Esc          Toggle start library; close overlays or exit",
        "Ctrl+Wheel / +/- UI zoom (Ctrl+0 resets to 100%)",
        "F11 / S / Ctrl+S Fullscreen; quick screenshot; Save PNG as...",
    };
    int y = box.y + cv.s(38);
    for (const char* line : rows) {
        cv.text_fit(box.x + cv.s(18), y, box.w - cv.s(36), line, pal::text, F_MONO);
        y += cv.s(24);
    }
}

}  // namespace

void VitroApp::render() {
    hits.clear();
    main_sv = {};
    side_sv = {};
    cv.clear(pal::bg);
    if (start_screen()) {
        draw_start_header(*this);
        draw_library(*this, {0, header_h(), cv.W, cv.H - header_h() - status_h()});
        draw_start_status(*this);
        draw_bars();
        if (st.help) draw_help(*this);
        return;
    }
    clamp();
    draw_header(*this);
    draw_sidebar(*this);
    Rect main = main_rect();
    switch (st.view) {
        case View::Tokens: draw_tokens(*this, main); break;
        case View::Attention: draw_attention(*this, main); break;
        case View::Activations: draw_activations(*this, main); break;
        case View::Spans: draw_spans(*this, main); break;
        case View::Net: draw_net(*this, main); break;
    }
    draw_status(*this);
    draw_bars();
    if (st.help) draw_help(*this);
    const Hit* h = hit_at(mx, my);
    if (h) describe_hit(*this, *h);
}

void VitroApp::move(int x, int y) {
    int dx = x - mx, dy = y - my;
    bool had = mx >= 0;
    mx = x;
    my = y;
    if (scroll_drag_ >= 0) {
        scroll_ref(scroll_drag_) = scrollbar_offset_at(cv, scroll_view(scroll_drag_), y);
        return;
    }
    if (!dragging) return;
    if (std::abs(x - press_x) + std::abs(y - press_y) > cv.s(4)) moved = true;
    if (st.view == View::Net && moved && had && main_rect().contains(press_x, press_y)) net_drag(*this, dx, dy);
}

void VitroApp::press(int x, int y) {
    mx = x; my = y;
    const Hit* h = hit_at(x, y);
    if (h && h->kind == H_SCROLL) {
        scroll_drag_ = h->a;
        scroll_ref(scroll_drag_) = scrollbar_offset_at(cv, scroll_view(scroll_drag_), y);
        return;
    }
    dragging = true;
    moved = false;
    press_x = x;
    press_y = y;
}

void VitroApp::release(int x, int y) {
    mx = x; my = y;
    if (scroll_drag_ >= 0) { scroll_drag_ = -1; return; }
    bool was_click = dragging && !moved;
    dragging = false;
    moved = false;
    if (was_click) click(x, y);
}

void VitroApp::wheel(int x, int y, int delta, bool ctrl) {
    (void)ctrl;
    const int step = delta > 0 ? -cv.s(78) : cv.s(78);
    if (start_screen()) {
        library_scroll = std::max(0, library_scroll + step);
        return;
    }
    if (x < side_w() && side_sv.needed()) {
        st.side_scroll = std::clamp(st.side_scroll + step, 0, side_sv.max_offset());
        return;
    }
    if (st.view == View::Net && main_rect().contains(x, y)) {
        st.net_zoom = std::clamp(st.net_zoom * (delta > 0 ? 1.1f : 1.0f / 1.1f), 0.3f, 5.0f);
        return;
    }
    if (main_sv.needed()) {
        st.scroll = std::clamp(st.scroll + step, 0, main_sv.max_offset());
        return;
    }
    if (!U().attn_families.empty()) {
        st.layer = std::clamp(st.layer + (delta > 0 ? -1 : 1), -1, U().attn_families[static_cast<size_t>(st.attn_fam)].layers - 1);
    }
}

bool VitroApp::scroll_key(int vk) {
    if (!main_sv.needed()) return false;
    int& cur = scroll_ref(0);
    int page = std::max(cv.s(100), main_sv.area.h * 9 / 10);
    switch (vk) {
        case 0x21: cur = std::max(0, cur - page); return true;
        case 0x22: cur = std::min(main_sv.max_offset(), cur + page); return true;
        case 0x24: cur = 0; return true;
        case 0x23: cur = main_sv.max_offset(); return true;
        default: return false;
    }
}

void VitroApp::click(int x, int y) {
    const Hit* found = hit_at(x, y);
    if (!found) return;
    Hit h = *found;
    switch (h.kind) {
        case H_LIB_ITEM: open_file(library_items[static_cast<size_t>(h.a)].path); break;
        case H_LIB_OPEN:
            if (h.a == 0) request_ = CMD_OPEN;
            else if (d.loaded()) library = false;
            break;
        case H_TAB: st.view = static_cast<View>(h.a); st.scroll = 0; break;
        case H_SIDE_UNIT: select_unit(h.a); break;
        case H_TOKEN: case H_FREE_TOKEN: select_token(h.a); break;
        case H_SPAN_ROW:
            st.span_sel = h.a;
            if (h.b >= 0) { select_token(std::max(U().t0, h.b)); st.view = View::Tokens; }
            break;
        case H_ATTN_FAM: st.attn_fam = h.a; clamp(); break;
        case H_LAYER: st.layer = h.a; break;
        case H_HEAD: st.head = h.a; break;
        case H_GRID: st.grid = h.a != 0; break;
        case H_COLOR: st.color_mode = h.a; break;
        case H_OVERLAY: st.overlay = h.a != 0; break;
        case H_MAP_ATTN: {
            int row, col;
            cell_of(*this, h, &row, &col);
            const AttnFamily& af = U().attn_families[static_cast<size_t>(st.attn_fam)];
            select_token(std::clamp(af.row_offset + row, 0, U().n_tokens() - 1));
            break;
        }
        case H_GRID_CELL: st.layer = h.a; st.head = h.b; st.grid = false; break;
        case H_LATENT_SEL: st.latent_idx = h.a; st.latent_row = -1; clamp(); break;
        case H_LATENT_CELL: {
            int row, col;
            cell_of(*this, h, &row, &col);
            (void)col;
            st.latent_row = row;
            break;
        }
        case H_LATENT_COS: {
            int row, col;
            cell_of(*this, h, &row, &col);
            (void)col;
            st.latent_row = row;
            break;
        }
        case H_NET_VIEW: net_pick_click(*this, x, y); break;
        case H_NET_BTN:
            if (h.a == 0) st.net_play = !st.net_play;
            else if (h.a == 1) st.net_path = !st.net_path;
            else if (h.a == 2) { st.net_yaw = NET_YAW0; st.net_pitch = NET_PITCH0; st.net_zoom = 1.0f; }
            else st.net_k = h.a - 10;
            break;
        default: break;
    }
}

bool VitroApp::key(int vk, bool ctrl, bool shift) {
    (void)ctrl;
    if (vk == 0x70) { st.help = !st.help; return true; }
    if (start_screen()) {
        if (vk == 'B' && d.loaded()) { library = false; return true; }
        if (scroll_key(vk)) return true;
        return false;
    }
    if (vk == 'B') { library = true; library_dirty = true; return true; }
    if (vk >= '1' && vk <= '5') { st.view = static_cast<View>(vk - '1'); st.scroll = 0; return true; }
    if (vk == 'C') { st.color_mode = (st.color_mode + 1) % 4; return true; }
    if (vk == 'G') { st.grid = !st.grid; return true; }
    if (vk == 'F' && !U().attn_families.empty()) {
        st.attn_fam = (st.attn_fam + 1) % static_cast<int>(U().attn_families.size());
        clamp();
        return true;
    }
    if (vk == 0xDB) { select_unit(std::max(0, st.unit - 1)); return true; }
    if (vk == 0xDD) { select_unit(std::min(static_cast<int>(d.units.size()) - 1, st.unit + 1)); return true; }
    const UnitTrace& u = U();
    if (vk == 0x25) { select_token(std::max(u.t0, st.token < 0 ? u.t0 : st.token - 1)); return true; }
    if (vk == 0x27) { select_token(std::min(u.n_tokens() - 1, st.token < 0 ? u.t0 : st.token + 1)); return true; }
    if (vk == 0x26 || vk == 0x28) {
        if (st.view == View::Tokens) {
            int cur = st.token < 0 ? u.t0 : st.token;
            select_token(std::clamp(cur + (vk == 0x26 ? -st.per_row : st.per_row), u.t0, u.n_tokens() - 1));
            return true;
        }
        if (main_sv.needed()) {
            st.scroll = std::clamp(st.scroll + (vk == 0x26 ? -cv.s(40) : cv.s(40)), 0, main_sv.max_offset());
            return true;
        }
        return false;
    }
    if (vk == 0x21 || vk == 0x22 || vk == 0x24 || vk == 0x23) return scroll_key(vk);
    if (vk == 'L' && !u.attn_families.empty()) {
        int L = u.attn_families[static_cast<size_t>(st.attn_fam)].layers;
        st.layer = shift ? (st.layer <= -1 ? L - 1 : st.layer - 1) : (st.layer >= L - 1 ? -1 : st.layer + 1);
        return true;
    }
    if (vk == 'H' && !u.attn_families.empty()) {
        int H = u.attn_families[static_cast<size_t>(st.attn_fam)].heads;
        st.head = shift ? (st.head <= -1 ? H - 1 : st.head - 1) : (st.head >= H - 1 ? -1 : st.head + 1);
        return true;
    }
    if (vk == 'E') {
        int dir = shift ? -1 : 1;
        int cur = st.token < 0 ? (dir > 0 ? u.t0 - 1 : u.n_tokens()) : st.token;
        for (int j = cur + dir; j >= u.t0 && j < u.n_tokens(); j += dir)
            if (u.rank.iv(static_cast<size_t>(j)) > 1) { select_token(j); return true; }
        status = "No prediction errors in that direction";
        return false;
    }
    if (vk == 'D') {
        if (u.diverge >= 0 && u.diverge < u.n_tokens()) { select_token(u.diverge); return true; }
        status = u.ftok.empty() ? "No free-generation trace recorded" : "No divergence in this unit";
        return false;
    }
    if (vk == 0xBB || vk == 0x6B) { st.cell_w = std::min(48, (st.cell_w > 0 ? st.cell_w : 26) + 3); return true; }
    if (vk == 0xBD || vk == 0x6D) { st.cell_w = std::max(14, (st.cell_w > 0 ? st.cell_w : 26) - 3); return true; }
    if (vk == '0') { st.cell_w = 0; return true; }
    if (st.view == View::Net) {
        if (vk == 0x20) { st.net_play = !st.net_play; return true; }
        if (vk == 'P') { st.net_path = !st.net_path; return true; }
        if (vk == 'R') { st.net_yaw = NET_YAW0; st.net_pitch = NET_PITCH0; st.net_zoom = 1.0f; return true; }
        if (vk == 'K') {
            static const int ks[] = {1, 2, 3, 4, 6, 8};
            for (int i = 0; i < 6; ++i)
                if (ks[i] == st.net_k) { st.net_k = ks[(i + 1) % 6]; break; }
            return true;
        }
    }
    return false;
}

bool VitroApp::tick(double seconds) {
    bool ch = false;
    if (notice_left > 0.0) {
        notice_left -= seconds;
        if (notice_left <= 0.0) { notice_text.clear(); ch = true; }
    }
    if (d.loaded() && !library && st.view == View::Net && st.net_play) {
        st.net_time += seconds;
        ch = true;
    }
    return ch;
}

bool VitroApp::escape() {
    if (st.help) { st.help = false; return true; }
    if (library && d.loaded()) { library = false; return true; }
    return false;
}

std::string VitroApp::open_file(const std::string& path) {
    try {
        TraceDump fresh(path);
        d = std::move(fresh);
    } catch (const std::exception& e) {
        error = e.what();
        status = error;
        return error;
    }
    attn_cache_.clear();
    cos_cache_.clear();
    net.reset();
    st.unit = 0;
    st.token = -1;
    st.attn_fam = 0;
    st.layer = -1;
    st.head = -1;
    st.latent_idx = 0;
    st.latent_row = -1;
    st.span_sel = -1;
    st.scroll = 0;
    st.side_scroll = 0;
    clamp();
    library = false;
    error.clear();
    remember(path);
    status = "Opened " + file_name_of(path);
    return "";
}

std::string VitroApp::title() {
    return d.loaded() ? ("Vitro — " + d.title) : "Vitro";
}

void VitroApp::set_ui_zoom(float zoom) {
    zoom_ = std::clamp(zoom, 0.6f, 3.0f);
    cv.set_scale(base_scale_ * zoom_);
}

void VitroApp::load_recent() {
    recent = read_recent(recent_file);
    library_dirty = true;
}

void VitroApp::remember(std::string path) {
    recent.erase(std::remove(recent.begin(), recent.end(), path), recent.end());
    recent.insert(recent.begin(), path);
    if (recent.size() > 12) recent.resize(12);
    library_dirty = true;
    write_recent(recent_file, recent);
}

void VitroApp::scan_library() {
    library_items = vitro::scan_library(library_dir, "vtdump", recent);
    library_dirty = false;
}

enum ViewMenuCmd { C_LIB = CMD_APP_FIRST + 1, C_V0 = CMD_APP_FIRST + 10, C_HELP = CMD_APP_FIRST + 30 };

std::vector<Menu> VitroApp::menus() {
    std::vector<Menu> m;
    m.push_back({"File", {{CMD_OPEN, "Open Trace...\tCtrl+O", false}, {C_LIB, "Library\tB", false}, {CMD_SAVE_PNG, "Save PNG...\tCtrl+S", false}, {0, "", false}, {CMD_EXIT, "Exit\tQ", false}}});
    Menu vm{"View", {}};
    static const char* vnames[VIEW_COUNT] = {"1  Tokens", "2  Attention", "3  Activations", "4  Spans / Parts", "5  3D Network"};
    for (int k = 0; k < VIEW_COUNT; ++k) vm.items.push_back({C_V0 + k, vnames[k], static_cast<int>(st.view) == k});
    vm.items.push_back({0, "", false});
    vm.items.push_back({CMD_ZOOM_IN, "Zoom In\tCtrl++", false});
    vm.items.push_back({CMD_ZOOM_OUT, "Zoom Out\tCtrl+-", false});
    vm.items.push_back({CMD_ZOOM_RESET, "Reset Zoom\tCtrl+0", false});
    vm.items.push_back({CMD_FULLSCREEN, "Fullscreen\tF11", false});
    m.push_back(vm);
    m.push_back({"Help", {{C_HELP, "Controls\tF1", st.help}}});
    return m;
}

void VitroApp::command(int id) {
    if (id >= C_V0 && id < C_V0 + VIEW_COUNT) { st.view = static_cast<View>(id - C_V0); st.scroll = 0; return; }
    if (id == C_LIB) { library = true; library_dirty = true; }
    else if (id == C_HELP) { st.help = !st.help; }
}

std::string VitroApp::state_json() {
    char buf[900];
    if (!d.loaded()) {
        std::snprintf(buf, sizeof buf, "{\"loaded\":false,\"library\":%d,\"help\":%d,\"zoom\":%.3f,\"recent\":%d,\"error\":\"",
                      library ? 1 : 0, st.help ? 1 : 0, zoom_, static_cast<int>(recent.size()));
    } else {
        std::snprintf(buf, sizeof buf,
                      "{\"loaded\":true,\"uid\":\"%s\",\"view\":\"%s\",\"unit\":%d,\"token\":%d,\"attn_fam\":%d,\"layer\":%d,\"head\":%d,"
                      "\"grid\":%d,\"color\":%d,\"cell_w\":%d,\"yaw\":%.4f,\"pitch\":%.4f,\"net_zoom\":%.4f,\"zoom\":%.3f,\"library\":%d,\"help\":%d,\"error\":\"",
                      json_escape(d.uid).c_str(), view_name(st.view), st.unit, st.token, st.attn_fam, st.layer, st.head,
                      st.grid ? 1 : 0, st.color_mode, st.cell_w, st.net_yaw, st.net_pitch, st.net_zoom, zoom_, library ? 1 : 0, st.help ? 1 : 0);
    }
    return std::string(buf) + json_escape(error) + "\",\"status\":\"" + json_escape(status) + "\"}";
}

}  // namespace vitro
