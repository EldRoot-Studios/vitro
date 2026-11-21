// View 1, "Tokens": reference vs free-generated strip, user-labeled spans, surprisal/rank/entropy/delta coloring, and detail card.
#include <algorithm>
#include <cmath>
#include <iterator>
#include <vector>

#include "view_app.hpp"

namespace vitro {

namespace {

struct Strip {
    int cw = 0, cell_h = 0, free_h = 0, span_h = 0, gap_h = 0, row_h = 0, per_row = 1, rows = 0;
};

Strip plan_strip(const VitroApp& a, int n, int avail_w, int avail_h, bool has_spans) {
    const Canvas& cv = a.cv;
    static const int widths[] = {48, 38, 34, 28, 24, 20, 17};
    std::vector<int> tries = a.st.cell_w > 0 ? std::vector<int>{a.st.cell_w} : std::vector<int>(std::begin(widths), std::end(widths));
    Strip s;
    for (int c : tries) {
        s.cw = cv.s(c);
        s.cell_h = std::clamp(static_cast<int>(s.cw * 0.85), cv.s(14), cv.s(24));
        s.free_h = a.U().ftok.empty() ? 0 : (c >= 9 ? cv.s(12) : cv.s(8));
        s.span_h = has_spans ? cv.s(5) : 0;
        s.gap_h = cv.s(13);
        s.row_h = s.gap_h + s.span_h + s.cell_h + (s.free_h > 0 ? 1 + s.free_h : 0) + cv.s(6);
        s.per_row = std::max(1, avail_w / s.cw);
        s.rows = (n + s.per_row - 1) / s.per_row;
        if (s.rows * s.row_h <= avail_h) break;
    }
    return s;
}

void draw_legend(VitroApp& a, int x, int y, bool has_scalar) {
    Canvas& cv = a.cv;
    int w = cv.s(120), h = cv.s(9);
    cv.text(x, y - cv.s(2), "surprisal (bits):", pal::dim, F_TINY);
    int x0 = x + cv.text_width("surprisal (bits):", F_TINY) + cv.s(6);
    for (int i = 0; i < w; ++i) cv.fill({x0 + i, y, 1, h}, surprisal(10.0 * i / w));
    const struct { double bits; const char* t; } ticks[] = {{0, "0"}, {1, "1"}, {3, "3"}, {6, "6"}, {10, "10"}};
    for (const auto& t : ticks) cv.text(x0 + static_cast<int>(t.bits / 10 * w) - cv.s(3), y + h + cv.s(1), t.t, pal::dim, F_TINY);
    int x_next = x0 + w + cv.s(26);
    if (has_scalar) {
        cv.text(x_next, y - cv.s(2), "scalar delta:", pal::dim, F_TINY);
        int x2 = x_next + cv.text_width("scalar delta:", F_TINY) + cv.s(6);
        for (int i = 0; i < w; ++i) cv.fill({x2 + i, y, 1, h}, delta_error(15.0 * i / w, a.d.delta_red_at));
        x_next = x2 + w + cv.s(26);
    }
    cv.frame({x_next, y, cv.s(14), h}, rgb(0xff5c5c));
    cv.text(x_next + cv.s(19), y - cv.s(2), "top-1 != target", pal::dim, F_TINY);
}

void draw_token_detail(VitroApp& a, Rect r) {
    Canvas& cv = a.cv;
    const TraceDump& d = a.d;
    const UnitTrace& u = a.U();
    a.panel(r, "Token Inspection");
    int j = a.st.token;
    const Hit* hv = a.hit_at(a.mx, a.my);
    if (hv && (hv->kind == H_TOKEN || hv->kind == H_FREE_TOKEN)) j = hv->a;
    if (j < u.t0) {
        cv.text_fit(r.x + cv.s(12), r.y + cv.s(34), r.w - cv.s(24),
                    "Click any token or use Left/Right arrows, E (errors), D (divergence).", pal::dim, F_UI);
        return;
    }
    size_t jj = static_cast<size_t>(j);
    float p = u.p.f(jj);
    int rank = u.rank.iv(jj);
    int lh = cv.s(19), x1 = r.x + cv.s(12), y = r.y + cv.s(26);
    cv.text(x1, y, fmt("token %d   \"%s\"", j, a.token_text(u, j).c_str()), pal::text, F_BIG);
    y += cv.s(30);
    auto line = [&](const std::string& t, Color c = pal::text) { cv.text_fit(x1, y, cv.s(450), t, c, F_UI); y += lh; };
    std::string tag_info;
    if (!u.token_tags.empty()) {
        const std::string& tg = d.tag_name(u.token_tags.iv(jj));
        if (!tg.empty()) tag_info = "tag: " + tg;
    }
    auto cov = a.spans_covering(j);
    if (!cov.empty()) {
        if (!tag_info.empty()) tag_info += "  |  ";
        tag_info += "spans: ";
        for (size_t k = 0; k < cov.size(); ++k) {
            if (k) tag_info += " > ";
            tag_info += cov[k]->label;
        }
    }
    if (!tag_info.empty()) line(tag_info, pal::dim);
    int allowed = u.n_allowed.empty() ? static_cast<int>(d.labels.size()) : u.n_allowed.iv(jj);
    line(fmt("target prob: %.4f (%.2f bits)  |  rank %d / %d", p, -std::log2(std::max(p, 1e-30f)), rank, allowed),
         rank > 1 ? pal::bad : pal::text);
    line(fmt("distribution entropy: %.2f bits", u.entropy.f(jj)));
    if (a.is_scalar_token(u, j)) {
        int top = u.top_id.iv(jj * static_cast<size_t>(d.top_k));
        double vt = d.value_of(u.tok.iv(jj));
        double vp = d.has_value(top) ? d.value_of(top) : u.val_mean.f(jj);
        line(fmt("scalar value: target %.4g, top %.4g (delta %.3g)", vt, vp, std::fabs(vp - vt)),
             std::fabs(vp - vt) >= d.delta_red_at ? pal::bad : pal::text);
        line(fmt("mean +/- std: %.2f +/- %.2f", u.val_mean.f(jj), u.val_std.f(jj)), pal::dim);
    }
    if (j < u.free_n()) {
        int ft = u.ftok.iv(jj);
        bool same = ft == u.tok.iv(jj);
        line(fmt("free rollout: \"%s\" (p=%.3f) - %s", d.label(ft).c_str(), u.fp.f(jj), same ? "matches target" : "differs"),
             same ? pal::good : pal::bad);
    }
    // Top-K candidates
    int x2 = r.x + cv.s(490), y2 = r.y + cv.s(30);
    cv.text(x2, y2, fmt("Top-%d Candidates", d.top_k), pal::dim, F_SMALL);
    y2 += cv.s(22);
    int target_id = u.tok.iv(jj);
    bool target_shown = false;
    for (int k = 0; k < d.top_k; ++k) {
        int cid = u.top_id.iv(jj * static_cast<size_t>(d.top_k) + static_cast<size_t>(k));
        if (cid < 0) break;
        float cp = u.top_p.f(jj * static_cast<size_t>(d.top_k) + static_cast<size_t>(k));
        bool is_target = cid == target_id;
        target_shown |= is_target;
        cv.text(x2, y2, d.label(cid), is_target ? pal::good : pal::text, F_MONO, cv.s(96), 2);
        a.bar({x2 + cv.s(104), y2 + cv.s(3), cv.s(160), cv.s(10)}, cp, is_target ? pal::good : pal::accent);
        cv.text(x2 + cv.s(272), y2, fmt("%.3f", cp), pal::dim, F_SMALL);
        y2 += cv.s(20);
    }
    if (!target_shown) {
        cv.text(x2, y2 + cv.s(2), d.label(target_id), pal::bad, F_MONO, cv.s(96), 2);
        cv.text(x2 + cv.s(104), y2 + cv.s(2), fmt("%.4f   target rank %d", p, rank), pal::bad, F_SMALL);
    }
}

}  // namespace

void draw_tokens(VitroApp& a, Rect area) {
    Canvas& cv = a.cv;
    const UnitTrace& u = a.U();
    Rect r = area.inset(cv.s(10));
    int detail_h = cv.s(216), strip_h = r.h - detail_h - cv.s(10);
    Rect box = {r.x, r.y, r.w, strip_h};
    a.panel(box, "");
    int wrong_n = 0;
    for (int j = u.t0; j < u.n_tokens(); ++j)
        if (u.rank.iv(static_cast<size_t>(j)) > 1) ++wrong_n;
    cv.text_fit(box.x + cv.s(12), box.y + cv.s(8), box.w - cv.s(360),
                fmt("%s - %d tokens (%d predicted), %d top-1 errors, %.1f total bits", u.label.c_str(), u.n_tokens(),
                    std::max(0, u.n_tokens() - u.t0), wrong_n, u.bits),
                pal::text, F_BOLD);
    static const char* modes[4] = {"delta / bits", "surprisal", "rank", "entropy"};
    const int mx0 = box.right() - cv.s(348), mw = cv.s(82);
    for (int k = 0; k < 4; ++k)
        a.button({mx0 + k * (mw + cv.s(4)), box.y + cv.s(6), mw, cv.s(22)}, modes[k], a.st.color_mode == k, H_COLOR, k, 0, F_TINY);

    draw_legend(a, box.x + cv.s(12), box.y + cv.s(34), !u.val_mean.empty());
    bool has_spans = !a.d.spans.empty();
    int pad_x = cv.s(12), top_y = box.y + cv.s(62);
    Strip stp = plan_strip(a, u.n_tokens(), box.w - pad_x * 2 - cv.s(12), box.bottom() - top_y - cv.s(8), has_spans);
    a.st.per_row = stp.per_row;

    Rect list_rect = {box.x + pad_x, top_y, box.w - pad_x * 2 - cv.s(10), box.bottom() - top_y - cv.s(8)};
    a.main_sv = {list_rect, stp.rows * stp.row_h};
    a.st.scroll = std::clamp(a.st.scroll, 0, a.main_sv.max_offset());
    cv.push_clip(list_rect);

    int row_offset_y = list_rect.y - a.st.scroll;
    for (int j = 0; j < u.n_tokens(); ++j) {
        int row = j / stp.per_row, col = j % stp.per_row;
        int x = list_rect.x + col * stp.cw;
        int y = row_offset_y + row * stp.row_h;
        if (y + stp.row_h < list_rect.y || y > list_rect.bottom()) continue;
        if (col == 0) cv.text(x, y, fmt("%d", j), pal::faint, F_TINY);
        int cy = y + stp.gap_h;
        if (has_spans) {
            auto cov = a.spans_covering(j);
            if (!cov.empty()) {
                const SpanInfo* sp = cov.back();
                Color scol = sp->tint.empty() ? pal::accent : parse_tint(sp->tint, pal::accent);
                cv.fill({x + 1, cy - cv.s(1), stp.cw - 2, cv.s(3)}, scol);
            }
            cy += stp.span_h;
        }
        bool wrong = false;
        Color c = a.cell_color(u, j, &wrong);
        Rect cell = {x + 1, cy, stp.cw - 2, stp.cell_h};
        cv.fill(cell, c);
        if (wrong) cv.frame(cell, rgb(0xff5c5c));
        std::string txt = a.token_text(u, j);
        FontId tf = cv.text_width(txt, F_SMALL) <= cell.w - 4 ? F_SMALL : F_TINY;
        cv.text_fit(cell.x + 2, cell.y + (cell.h - cv.fonts.height(tf)) / 2, cell.w - 4, txt, text_on(c), tf, 1);
        a.add_hit(cell.intersect(list_rect), H_TOKEN, j);

        if (stp.free_h > 0 && j >= u.t0) {
            Rect fc = {x + 1, cell.bottom() + 1, stp.cw - 2, stp.free_h};
            if (j < u.free_n()) {
                bool same = u.ftok.iv(static_cast<size_t>(j)) == u.tok.iv(static_cast<size_t>(j));
                Color fcol = same ? rgb(0x275a40) : rgb(0x7a2a36);
                cv.fill(fc, fcol);
                std::string ftxt = a.free_text(u, j);
                cv.text_fit(fc.x + 1, fc.y + 1, fc.w - 2, ftxt, pal::text, F_TINY, 1);
            } else {
                cv.fill(fc, pal::panel2);
            }
            a.add_hit(fc.intersect(list_rect), H_FREE_TOKEN, j);
        }
        if (j == a.st.token) cv.frame(cell.inset(-1), pal::select, 2);
        if (j == u.diverge) cv.text(x + 1, y + cv.s(1), "div", pal::warn, F_TINY);
    }
    cv.pop_clip();
    draw_token_detail(a, {r.x, box.bottom() + cv.s(10), r.w, detail_h});
}

}  // namespace vitro
