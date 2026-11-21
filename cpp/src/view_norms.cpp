// View 3, "Activations": residual/sublayer L2 norms across layers and optional 2D latent state/similarity maps.
#include <algorithm>
#include <cmath>

#include "view_app.hpp"

namespace vitro {

namespace {

void draw_norm_panel(VitroApp& a, Rect r, const UnitTrace& u) {
    Canvas& cv = a.cv;
    const NormBlock& nb = u.norms;
    a.panel(r, "");
    if (nb.data.empty()) {
        cv.text(r.x + cv.s(12), r.y + cv.s(30), "No per-layer activation norms recorded.", pal::dim, F_UI);
        return;
    }
    const int L = static_cast<int>(nb.data.dim(0));
    const int C = static_cast<int>(nb.data.dim(1));
    const int R = static_cast<int>(nb.data.dim(2));
    cv.text(r.x + cv.s(10), r.y + cv.s(6), fmt("%s (%d layers x %d components x %d tokens)", nb.name.c_str(), L, C, R), pal::text, F_BOLD);

    float vmax = 1e-6f;
    for (size_t i = 0; i < nb.data.n; ++i) vmax = std::max(vmax, nb.data.f(i));

    int x0 = r.x + cv.s(86), top = r.y + cv.s(36), avail_w = r.w - cv.s(104), avail_h = r.h - cv.s(68);
    int row_h = std::max(cv.s(18), std::min(cv.s(42), avail_h / std::max(1, L)));
    int sub_h = std::max(2, (row_h - cv.s(4)) / std::max(1, C));
    static const Color comp_cols[] = {rgb(0x7b6df0), rgb(0x4fb0c6), rgb(0xb06cf0), rgb(0xf0b43c)};

    for (int l = 0; l < L; ++l) {
        int y = top + l * row_h;
        std::string ln = l < static_cast<int>(nb.layers.size()) ? nb.layers[static_cast<size_t>(l)] : fmt("L%d", l + 1);
        cv.text_fit(r.x + cv.s(8), y + row_h / 2 - cv.s(6), cv.s(74), ln, pal::dim, F_SMALL);
        for (int c = 0; c < C; ++c) {
            Rect band = {x0, y + c * sub_h, avail_w, std::max(2, sub_h - 1)};
            cv.heat(band, 1, R, [&](int, int pos) {
                return nb.data.f((static_cast<size_t>(l) * static_cast<size_t>(C) + static_cast<size_t>(c)) * static_cast<size_t>(R) + static_cast<size_t>(pos)) / vmax;
            }, [&](float t) {
                return mix(rgb(0x161a20), comp_cols[static_cast<size_t>(c) % 4], std::sqrt(std::clamp(t, 0.0f, 1.0f)));
            });
            cv.frame(band, pal::border);
        }
    }
    if (a.st.token >= nb.row_offset && a.st.token < nb.row_offset + R) {
        int col = a.st.token - nb.row_offset;
        int sx = x0 + static_cast<int>((static_cast<long long>(col) * 2 + 1) * avail_w / (2 * R));
        cv.vline(sx, top - cv.s(2), top + L * row_h, pal::select);
    }
    int lx = r.x + cv.s(12), ly = r.bottom() - cv.s(24);
    for (int c = 0; c < C; ++c) {
        Color col = comp_cols[static_cast<size_t>(c) % 4];
        cv.fill({lx, ly + cv.s(3), cv.s(12), cv.s(10)}, col);
        std::string cn = c < static_cast<int>(nb.components.size()) ? nb.components[static_cast<size_t>(c)] : fmt("comp %d", c);
        lx += cv.s(16) + cv.text(lx + cv.s(16), ly, cn, pal::dim, F_SMALL) + cv.s(16);
    }
}

void draw_latent_panel(VitroApp& a, Rect r, const UnitTrace& u) {
    Canvas& cv = a.cv;
    a.panel(r, "");
    const LatentMap& lm = u.latents[static_cast<size_t>(a.st.latent_idx)];
    int top_pad = cv.s(34);
    int bx = r.x + cv.s(12);
    for (size_t i = 0; i < u.latents.size(); ++i) {
        int bw = std::max(cv.s(90), cv.text_width(u.latents[i].name, F_SMALL) + cv.s(16));
        a.button({bx, r.y + cv.s(6), bw, cv.s(22)}, u.latents[i].name, a.st.latent_idx == static_cast<int>(i), H_LATENT_SEL, static_cast<int>(i), 0, F_SMALL);
        bx += bw + cv.s(6);
    }
    const int S = static_cast<int>(lm.data.dim(0)), D = static_cast<int>(lm.data.dim(1));
    float vmax = 1e-6f;
    for (size_t i = 0; i < lm.data.n; ++i) vmax = std::max(vmax, std::fabs(lm.data.f(i)));
    int left_w = (r.w - cv.s(30)) * 6 / 10;
    Rect map = {r.x + cv.s(54), r.y + top_pad + cv.s(18), std::max(40, left_w - cv.s(60)), std::max(40, r.h - top_pad - cv.s(46))};
    cv.heat(map, S, D, [&](int row, int col) { return lm.data.f(static_cast<size_t>(row) * D + col) / vmax; },
            [](float v) { return diverging(v); });
    cv.frame(map.inset(-1), pal::border);
    a.add_hit(map, H_LATENT_CELL, S, D);
    cv.text(map.x, map.y - cv.s(16), fmt("%s (%d x %d, max |x| %.2f)", lm.col_label.c_str(), S, D, vmax), pal::dim, F_TINY);

    int side_x = map.right() + cv.s(16);
    int cos_sz = std::min(r.right() - side_x - cv.s(12), map.h);
    Rect cmap = {side_x, map.y, cos_sz, cos_sz};
    const std::vector<float>& cos = a.latent_cos(a.st.unit, a.st.latent_idx);
    cv.heat(cmap, S, S, [&](int row, int col) { return cos[static_cast<size_t>(row) * S + col]; }, [](float v) { return similarity(v); });
    cv.frame(cmap.inset(-1), pal::border);
    a.add_hit(cmap, H_LATENT_COS, S, S);
    cv.text(cmap.x, cmap.y - cv.s(16), fmt("%s cosine similarity", lm.row_label.c_str()), pal::dim, F_TINY);
}

}  // namespace

void draw_activations(VitroApp& a, Rect area) {
    Rect r = area.inset(a.cv.s(10));
    const UnitTrace& u = a.U();
    if (u.latents.empty()) {
        draw_norm_panel(a, r, u);
        return;
    }
    int top_h = u.norms.data.empty() ? 0 : std::max(a.cv.s(180), r.h * 45 / 100);
    if (top_h > 0) draw_norm_panel(a, {r.x, r.y, r.w, top_h}, u);
    int gap = top_h > 0 ? a.cv.s(10) : 0;
    draw_latent_panel(a, {r.x, r.y + top_h + gap, r.w, r.h - top_h - gap}, u);
}

bool describe_activations(VitroApp& a, const Hit& h) {
    const UnitTrace& u = a.U();
    if (u.latents.empty()) return false;
    const LatentMap& lm = u.latents[static_cast<size_t>(a.st.latent_idx)];
    if (h.kind == H_LATENT_CELL) {
        int r, c;
        cell_of(a, h, &r, &c);
        a.status = fmt("%s %d, %s %d: %.4f", lm.row_label.c_str(), r, lm.col_label.c_str(), c, lm.data.f(static_cast<size_t>(r) * h.b + c));
        return true;
    }
    if (h.kind == H_LATENT_COS) {
        int r, c;
        cell_of(a, h, &r, &c);
        a.status = fmt("cosine(%s %d, %s %d) = %.3f", lm.row_label.c_str(), r, lm.row_label.c_str(), c,
                       a.latent_cos(a.st.unit, a.st.latent_idx)[static_cast<size_t>(r) * h.b + c]);
        return true;
    }
    return false;
}

}  // namespace vitro
