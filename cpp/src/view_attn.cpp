// View 2, "Attention": dynamic inspector across any recorded attention family (self, cross, encoder, etc.).
#include <algorithm>
#include <cmath>

#include "view_app.hpp"

namespace vitro {

namespace {

std::string layer_text(int layer) { return layer < 0 ? "layer mean" : fmt("layer %d", layer + 1); }
std::string head_text(int head) { return head < 0 ? "head mean" : fmt("head %d", head + 1); }

void controls(VitroApp& a, Rect r, const UnitTrace& u, const AttnFamily& af) {
    Canvas& cv = a.cv;
    a.panel(r, "Attention Inspector");
    int x = r.x + cv.s(10), w = r.w - cv.s(20), y = r.y + cv.s(30);
    cv.text_fit(x, y, w, u.label, pal::text, F_SMALL);
    y += cv.s(16);
    cv.text_fit(x, y, w, fmt("%d rows x %d cols", af.rows, af.cols), pal::dim, F_SMALL);
    y += cv.s(24);
    cv.text(x, y, "Attention Group", pal::dim, F_SMALL);
    y += cv.s(18);
    for (size_t i = 0; i < u.attn_families.size(); ++i) {
        a.button({x, y, w, cv.s(24)}, u.attn_families[i].name, a.st.attn_fam == static_cast<int>(i), H_ATTN_FAM, static_cast<int>(i), 0, F_SMALL);
        y += cv.s(28);
    }
    y += cv.s(8);
    cv.text(x, y, "Layer", pal::dim, F_SMALL);
    y += cv.s(18);
    int bw = (w - cv.s(8)) / 3;
    for (int l = 0; l < af.layers; ++l)
        a.button({x + (l % 3) * (bw + cv.s(4)), y + (l / 3) * cv.s(28), bw, cv.s(24)}, fmt("%d", l + 1), a.st.layer == l, H_LAYER, l);
    y += cv.s(28) * ((af.layers + 2) / 3);
    a.button({x, y, w, cv.s(24)}, "mean across layers", a.st.layer < 0, H_LAYER, -1, 0, F_SMALL);
    y += cv.s(34);
    cv.text(x, y, "Head", pal::dim, F_SMALL);
    y += cv.s(18);
    for (int h = 0; h < af.heads; ++h)
        a.button({x + (h % 3) * (bw + cv.s(4)), y + (h / 3) * cv.s(28), bw, cv.s(24)}, fmt("%d", h + 1), a.st.head == h, H_HEAD, h);
    y += cv.s(28) * ((af.heads + 2) / 3);
    a.button({x, y, w, cv.s(24)}, "mean across heads", a.st.head < 0, H_HEAD, -1, 0, F_SMALL);
    y += cv.s(36);
    a.button({x, y, w, cv.s(28)}, a.st.grid ? "single heatmap" : "all layers x heads", a.st.grid, H_GRID, a.st.grid ? 0 : 1, 0, F_SMALL);
}

void row_detail(VitroApp& a, Rect r, const UnitTrace& u, const AttnFamily& af, int row) {
    Canvas& cv = a.cv;
    a.panel(r, "Row Weights");
    int x = r.x + cv.s(10), y = r.y + cv.s(28);
    if (row < 0 || row >= af.rows) {
        cv.text_fit(x, y, r.w - cv.s(20), "Click a heatmap row or select a token in View 1.", pal::dim, F_SMALL);
        return;
    }
    int j = af.row_offset + row;
    cv.text_fit(x, y, r.w - cv.s(20), fmt("%s %d \"%s\"", af.row_label.c_str(), j, a.token_text(u, j).c_str()), pal::text, F_BOLD);
    y += cv.s(20);
    cv.text_fit(x, y, r.w - cv.s(20), fmt("%s, %s", layer_text(a.st.layer).c_str(), head_text(a.st.head).c_str()), pal::dim, F_SMALL);
    y += cv.s(24);
    const std::vector<float>& m = a.attn_map(a.st.unit, a.st.attn_fam, a.st.layer, a.st.head);
    const float* v = &m[static_cast<size_t>(row) * static_cast<size_t>(af.cols)];
    std::vector<int> order(static_cast<size_t>(af.cols));
    for (int c = 0; c < af.cols; ++c) order[static_cast<size_t>(c)] = c;
    int top_n = std::min(af.cols, std::max(1, (r.bottom() - y - cv.s(18)) / cv.s(19)));
    std::partial_sort(order.begin(), order.begin() + top_n, order.end(), [&](int p, int q) { return v[p] > v[q]; });
    float vmax = std::max(1e-9f, v[order[0]]);
    for (int k = 0; k < top_n; ++k) {
        int q = order[static_cast<size_t>(k)];
        std::string clab = !af.causal && af.col_label != af.row_label && u.src_tok.empty() ? fmt("c%d", q) : a.source_text(u, q);
        if (af.cols == u.n_source() && !u.src_tok.empty()) clab = a.source_text(u, q);
        else if (af.cols <= u.n_tokens() && af.causal) clab = a.token_text(u, q);
        cv.text(x, y, fmt("%d", q), pal::dim, F_TINY, cv.s(30), 2);
        cv.text_fit(x + cv.s(34), y, cv.s(72), clab, pal::text, F_MONO);
        a.bar({x + cv.s(110), y + 2, r.w - cv.s(190), cv.s(11)}, v[q] / vmax, pal::accent);
        cv.text(x + r.w - cv.s(74), y, fmt("%.3f", v[q]), pal::dim, F_TINY);
        y += cv.s(19);
    }
}

void single_map(VitroApp& a, Rect area, const UnitTrace& u, const AttnFamily& af) {
    Canvas& cv = a.cv;
    int R = af.rows, C = af.cols, detail_w = cv.s(330);
    Rect mapbox = {area.x, area.y, area.w - detail_w - cv.s(10), area.h};
    a.panel(mapbox, "");
    cv.text_fit(mapbox.x + cv.s(10), mapbox.y + cv.s(6), mapbox.w - cv.s(20),
                fmt("%s - %s, %s", af.name.c_str(), layer_text(a.st.layer).c_str(), head_text(a.st.head).c_str()),
                pal::text, F_BOLD);
    int left = cv.s(54), top = cv.s(52), right = cv.s(14), bottom = cv.s(16);
    int avail_w = mapbox.w - left - right, avail_h = mapbox.h - top - bottom;
    int col_w = C <= 48 ? std::clamp(avail_w / std::max(1, C), 3, cv.s(26)) : std::max(1, avail_w / std::max(1, C));
    int map_w = C * col_w <= avail_w ? C * col_w : avail_w;
    int row_h = std::clamp(avail_h / std::max(1, R), 1, cv.s(14));
    int map_h = R * row_h <= avail_h ? R * row_h : avail_h;
    Rect map = {mapbox.x + left, mapbox.y + top, map_w, map_h};
    const std::vector<float>& m = a.attn_map(a.st.unit, a.st.attn_fam, a.st.layer, a.st.head);
    float vmax = 1e-9f;
    for (float v : m) vmax = std::max(vmax, v);
    cv.heat(map, R, C, [&](int r, int c) { return m[static_cast<size_t>(r) * C + c]; },
            [&](float v) { return magma(std::pow(std::clamp(v / vmax, 0.0f, 1.0f), 0.55f)); });
    cv.frame(map.inset(-1), pal::border);
    a.add_hit(map, H_MAP_ATTN, R, C);
    int row_every = std::max(1, cv.s(30) / std::max(1, map.h / std::max(1, R)));
    for (int r = 0; r < R; r += row_every) {
        int y = map.y + static_cast<int>(static_cast<long long>(r) * map.h / R);
        cv.text(map.x - cv.s(44), y - cv.s(1), fmt("%d", af.row_offset + r), pal::faint, F_TINY, cv.s(40), 2);
    }
    int tick = C <= 32 ? 4 : C <= 120 ? 10 : C <= 300 ? 25 : 100;
    for (int c = 0; c < C; c += tick) {
        int x = map.x + static_cast<int>(static_cast<long long>(c) * map.w / C);
        cv.vline(x, map.y - cv.s(4), map.y - 1, pal::faint);
        cv.text(x - cv.s(2), map.y - cv.s(18), fmt("%d", c), pal::faint, F_TINY);
    }
    int sel_row = -1;
    if (a.st.token >= af.row_offset && a.st.token < af.row_offset + R) sel_row = a.st.token - af.row_offset;
    if (sel_row >= 0) {
        int y = map.y + static_cast<int>((static_cast<long long>(sel_row) * 2 + 1) * map.h / (2 * R));
        cv.hline(map.x - cv.s(6), map.right(), y, pal::select);
    }
    row_detail(a, {mapbox.right() + cv.s(10), area.y, detail_w, area.h}, u, af, sel_row);
}

void grid_maps(VitroApp& a, Rect area, const AttnFamily& af) {
    Canvas& cv = a.cv;
    a.panel(area, "");
    cv.text_fit(area.x + cv.s(10), area.y + cv.s(6), area.w - cv.s(20),
                fmt("%s: layers (rows) x heads (cols), last row/col = mean. Click cell to zoom.", af.name.c_str()),
                pal::text, F_BOLD);
    int R = af.rows, C = af.cols, layers = af.layers, heads = af.heads;
    int left = cv.s(64), top = cv.s(50), gap = cv.s(4);
    int cw = (area.w - left - cv.s(12) - gap * heads) / (heads + 1);
    int ch = (area.h - top - cv.s(12) - gap * layers) / (layers + 1);
    for (int l = 0; l <= layers; ++l) {
        int ly = l == layers ? -1 : l;
        int y = area.y + top + l * (ch + gap);
        cv.text(area.x + cv.s(8), y + ch / 2 - cv.s(6), ly < 0 ? "mean" : fmt("L%d", l + 1), pal::dim, F_SMALL);
        for (int h = 0; h <= heads; ++h) {
            int hy = h == heads ? -1 : h;
            int x = area.x + left + h * (cw + gap);
            if (l == 0) cv.text(x, area.y + top - cv.s(18), hy < 0 ? "mean" : fmt("H%d", h + 1), pal::dim, F_SMALL, cw, 1);
            Rect cell = {x, y, cw, ch};
            const std::vector<float>& m = a.attn_map(a.st.unit, a.st.attn_fam, ly, hy);
            float vmax = 1e-9f;
            for (float v : m) vmax = std::max(vmax, v);
            cv.heat(cell, R, C, [&](int r, int c) { return m[static_cast<size_t>(r) * C + c]; },
                    [&](float v) { return magma(std::pow(std::clamp(v / vmax, 0.0f, 1.0f), 0.55f)); });
            bool cur = ly == a.st.layer && hy == a.st.head;
            cv.frame(cell.inset(-1), cur ? pal::select : pal::border, cur ? 2 : 1);
            a.add_hit(cell, H_GRID_CELL, ly, hy);
        }
    }
}

}  // namespace

void draw_attention(VitroApp& a, Rect area) {
    Rect r = area.inset(a.cv.s(10));
    const UnitTrace& u = a.U();
    if (u.attn_families.empty()) {
        a.panel(r, "Attention Inspector");
        a.cv.text(r.x + a.cv.s(16), r.y + a.cv.s(40), "No attention matrices were recorded for this unit.", pal::dim, F_UI);
        return;
    }
    const AttnFamily& af = u.attn_families[static_cast<size_t>(a.st.attn_fam)];
    int cw = a.cv.s(220);
    controls(a, {r.x, r.y, cw, r.h}, u, af);
    Rect main = {r.x + cw + a.cv.s(10), r.y, r.w - cw - a.cv.s(10), r.h};
    if (a.st.grid) grid_maps(a, main, af);
    else single_map(a, main, u, af);
}

bool describe_attention(VitroApp& a, const Hit& h) {
    const UnitTrace& u = a.U();
    if (u.attn_families.empty()) return false;
    const AttnFamily& af = u.attn_families[static_cast<size_t>(a.st.attn_fam)];
    switch (h.kind) {
        case H_ATTN_FAM: a.status = "switch attention map to " + u.attn_families[static_cast<size_t>(h.a)].name; return true;
        case H_MAP_ATTN: {
            int row, col;
            cell_of(a, h, &row, &col);
            float w = a.attn_map(a.st.unit, a.st.attn_fam, a.st.layer, a.st.head)[static_cast<size_t>(row) * h.b + col];
            a.status = fmt("query %d attends to key %d with weight %.4f", af.row_offset + row, col, w);
            return true;
        }
        case H_GRID_CELL: a.status = fmt("%s, %s - click to focus", layer_text(h.a).c_str(), head_text(h.b).c_str()); return true;
        case H_LAYER: a.status = layer_text(h.a); return true;
        case H_HEAD: a.status = head_text(h.a); return true;
        default: return false;
    }
}

}  // namespace vitro
