// View 5, "3D Net": completely dynamic 3D neural pipeline built from StageDef + LinkDef arrays in the trace.
// Supports arbitrary model graphs (decoder-only, encoder-decoder, multi-branch latent networks), generalized backward
// attention rollout, golden importance paths, flowing wave particles, and user-defined stage/group labels.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

#include "view_app.hpp"

namespace vitro {

namespace {

constexpr double WAVE_PERIOD = 8.0;
constexpr float RHO = 0.5f;

struct NNode {
    float x = 0, y = 0, z = 0;
    int stage = 0, idx = 0;
    float act = 0, imp = 0;
    float sx = 0, sy = 0, scale = 1;
};

enum EdgeKind : uint8_t { E_RESID = 0, E_SELF, E_CROSS, E_OUT };

struct NEdge {
    int a = 0, b = 0;
    float w = 0, imp = 0;
    uint8_t kind = E_SELF;
    int stage = 0;
    Color tint = rgb(0x7b6df0);
};

struct StageGroupSpan {
    int a = 0, b = 0;
    std::string label;
};

struct Net {
    int unit = -1, token = -1, head = -2, k = 0;
    std::vector<NNode> nodes;
    std::vector<NEdge> edges;
    std::vector<int> first, count;
    std::vector<float> stage_x;
    std::vector<StageGroupSpan> groups;
    int nstages = 0, s_pred = 0, s_out = 0;
    int sel_p = 0, true_out = -1, out_first = 0;
    float half_span = 100.0f;
    std::vector<int> hot;
    bool cam_ok = false, box_ok = false;
    float cam_yaw = 0, cam_pitch = 0, cam_zoom = 0, box_yaw = 0, box_pitch = 0;
    Rect cam_view;
    Camera3D cam;
    float box_x0 = 0, box_x1 = 0, box_y0 = 0, box_y1 = 0;
    float lod = 1.0f;
    double last_ms = 0.0;

    int node(int stage, int i) const { return first[static_cast<size_t>(stage)] + i; }
};

int selected_token(const VitroApp& a, const UnitTrace& u) {
    if (a.st.token >= u.t0 && a.st.token < u.n_tokens()) return a.st.token;
    return std::min(u.t0, std::max(0, u.n_tokens() - 1));
}

std::vector<float> avg_link_weights(const LinkDef& lk, int head_sel) {
    const Arr& w = lk.weights;
    const bool has_h = w.shape.size() == 3;
    const int H = has_h ? static_cast<int>(w.dim(0)) : 1;
    const int R = static_cast<int>(w.dim(has_h ? 1 : 0));
    const int C = static_cast<int>(w.dim(has_h ? 2 : 1));
    std::vector<float> out(static_cast<size_t>(R) * static_cast<size_t>(C), 0.0f);
    int h0 = (head_sel >= 0 && head_sel < H) ? head_sel : 0;
    int h1 = (head_sel >= 0 && head_sel < H) ? head_sel + 1 : H;
    float inv = 1.0f / static_cast<float>(std::max(1, h1 - h0));
    size_t rc = static_cast<size_t>(R) * static_cast<size_t>(C);
    for (int h = h0; h < h1; ++h) {
        size_t base = static_cast<size_t>(h) * rc;
        for (size_t i = 0; i < rc; ++i) out[i] += w.f(base + i) * inv;
    }
    return out;
}

std::shared_ptr<Net> build_net(const VitroApp& a, const UnitTrace& u, int j_sel) {
    const TraceDump& d = a.d;
    auto net = std::make_shared<Net>();
    Net& n = *net;
    n.unit = a.st.unit; n.token = j_sel; n.head = a.st.head; n.k = a.st.net_k;
    const int ns = static_cast<int>(u.stages.size());
    n.nstages = ns + 1;                                             // user stages + dynamic output candidate stage
    n.s_out = ns;
    n.s_pred = std::max(0, ns - 1);

    n.stage_x.assign(static_cast<size_t>(n.nstages), 0.0f);
    float cur_x = 0.0f;
    for (int s = 0; s < ns; ++s) {
        if (s > 0) cur_x += 40.0f + u.stages[static_cast<size_t>(s)].gap_before;
        n.stage_x[static_cast<size_t>(s)] = cur_x;
    }
    n.stage_x[static_cast<size_t>(n.s_out)] = cur_x + 56.0f;
    n.half_span = n.stage_x.back() * 0.5f;
    n.first.assign(static_cast<size_t>(n.nstages), 0);
    n.count.assign(static_cast<size_t>(n.nstages), 0);

    for (int s = 0; s < ns; ++s) {
        const StageDef& st = u.stages[static_cast<size_t>(s)];
        float amax = 1e-6f;
        for (size_t i = 0; i < st.act.n; ++i) amax = std::max(amax, st.act.f(i));
        n.first[static_cast<size_t>(s)] = static_cast<int>(n.nodes.size());
        n.count[static_cast<size_t>(s)] = st.count;
        for (int i = 0; i < st.count; ++i) {
            NNode nd;
            nd.stage = s; nd.idx = i;
            nd.x = n.stage_x[static_cast<size_t>(s)] + (hash01(static_cast<uint32_t>(s), static_cast<uint32_t>(i) + 9000u) - 0.5f) * st.x_jitter;
            nd.y = (0.5f - (static_cast<float>(i) + 0.5f) / static_cast<float>(std::max(1, st.count))) * 2.0f * st.span_y;
            nd.z = (hash01(static_cast<uint32_t>(s), static_cast<uint32_t>(i)) - 0.5f) * 2.0f * st.z_jitter;
            nd.act = static_cast<size_t>(i) < st.act.n ? std::clamp(st.act.f(static_cast<size_t>(i)) / amax, 0.05f, 1.0f) : 0.35f;
            n.nodes.push_back(nd);
        }
    }

    // Output candidates column
    const size_t jj = static_cast<size_t>(std::clamp(j_sel, 0, std::max(0, u.n_tokens() - 1)));
    std::vector<std::pair<int, float>> outs;
    if (u.n_tokens() > 0 && j_sel >= u.t0) {
        int right = u.tok.iv(jj);
        bool listed = false;
        for (int k = 0; k < d.top_k; ++k) {
            int id = u.top_id.iv(jj * static_cast<size_t>(d.top_k) + static_cast<size_t>(k));
            if (id < 0) break;
            float pr = u.top_p.f(jj * static_cast<size_t>(d.top_k) + static_cast<size_t>(k));
            outs.emplace_back(id, pr);
            listed |= id == right;
        }
        if (!listed) outs.emplace_back(right, u.p.f(jj));
    }
    n.out_first = static_cast<int>(n.nodes.size());
    n.first[static_cast<size_t>(n.s_out)] = n.out_first;
    n.count[static_cast<size_t>(n.s_out)] = static_cast<int>(outs.size());
    int right_id = (u.n_tokens() > 0 && j_sel >= 0 && j_sel < u.n_tokens()) ? u.tok.iv(jj) : -1;
    for (size_t i = 0; i < outs.size(); ++i) {
        NNode nd;
        nd.stage = n.s_out; nd.idx = outs[i].first;
        nd.x = n.stage_x[static_cast<size_t>(n.s_out)];
        nd.y = (0.5f - (static_cast<float>(i) + 0.5f) / static_cast<float>(std::max<size_t>(1, outs.size()))) * 90.0f;
        nd.act = outs[i].second;
        n.nodes.push_back(nd);
        if (outs[i].first == right_id) n.true_out = n.out_first + static_cast<int>(i);
    }

    // Group brackets
    for (int s = 0; s < ns;) {
        const std::string& grp = u.stages[static_cast<size_t>(s)].group;
        int e = s;
        while (e + 1 < ns && u.stages[static_cast<size_t>(e + 1)].group == grp) ++e;
        n.groups.push_back({s, e, grp.empty() ? u.stages[static_cast<size_t>(s)].label : grp});
        s = e + 1;
    }
    n.groups.push_back({n.s_out, n.s_out, "Output"});

    // Generalized Attention Rollout over arbitrary DAG of stages & links
    std::vector<std::vector<float>> imp(static_cast<size_t>(n.nstages));
    for (int s = 0; s < n.nstages; ++s) imp[static_cast<size_t>(s)].assign(static_cast<size_t>(n.count[static_cast<size_t>(s)]), 0.0f);
    n.sel_p = ns > 0 ? std::clamp(j_sel - 1, 0, std::max(0, n.count[static_cast<size_t>(n.s_pred)] - 1)) : 0;
    if (ns > 0 && n.count[static_cast<size_t>(n.s_pred)] > 0) imp[static_cast<size_t>(n.s_pred)][static_cast<size_t>(n.sel_p)] = 1.0f;

    std::vector<float> edge_flow;
    auto push = [&](int a_node, int b_node, float w, uint8_t kind, int stage, float flow, Color tint) {
        n.edges.push_back({a_node, b_node, w, 0.0f, kind, stage, tint});
        edge_flow.push_back(flow);
    };

    if (ns > 0 && n.count[static_cast<size_t>(n.s_pred)] > 0) {
        for (int o = 0; o < n.count[static_cast<size_t>(n.s_out)]; ++o) {
            float pr = n.nodes[static_cast<size_t>(n.out_first + o)].act;
            push(n.node(n.s_pred, n.sel_p), n.out_first + o, pr, E_OUT, n.s_out, pr, rgb(0xffc94a));
        }
    }

    std::vector<int> idx, pick;
    const int K = a.st.net_k;
    for (int dst = ns - 1; dst >= 0; --dst) {
        std::vector<int> res_links, attn_links;
        for (size_t li = 0; li < u.links.size(); ++li) {
            if (u.links[li].dst != dst) continue;
            if (u.links[li].kind == "resid" || u.links[li].weights.empty()) res_links.push_back(static_cast<int>(li));
            else attn_links.push_back(static_cast<int>(li));
        }
        const int dst_cnt = n.count[static_cast<size_t>(dst)];
        const float res_share = res_links.empty() ? 0.0f : (attn_links.empty() ? 1.0f : RHO);
        const float attn_total_share = 1.0f - res_share;

        if (!res_links.empty()) {
            const float per_res = res_share / static_cast<float>(res_links.size());
            for (int li : res_links) {
                const int src = u.links[static_cast<size_t>(li)].src;
                const int src_cnt = n.count[static_cast<size_t>(src)];
                const int m = std::min(dst_cnt, src_cnt);
                for (int p = 0; p < m; ++p) {
                    float f = per_res * imp[static_cast<size_t>(dst)][static_cast<size_t>(p)];
                    imp[static_cast<size_t>(src)][static_cast<size_t>(p)] += f;
                    push(n.node(src, p), n.node(dst, p), 1.0f, E_RESID, dst, f, rgb(0x7b6df0));
                }
            }
        }
        if (!attn_links.empty()) {
            const float per_attn = attn_total_share / static_cast<float>(attn_links.size());
            for (int li : attn_links) {
                const LinkDef& lk = u.links[static_cast<size_t>(li)];
                const int src = lk.src;
                const int src_cnt = n.count[static_cast<size_t>(src)];
                const bool has_h = lk.weights.shape.size() == 3;
                const int R = static_cast<int>(lk.weights.dim(has_h ? 1 : 0));
                const int C = static_cast<int>(lk.weights.dim(has_h ? 2 : 1));
                std::vector<float> mat = avg_link_weights(lk, a.st.head);
                const Color col = parse_tint(lk.tint, lk.kind == "cross" ? rgb(0xb06cf0) : rgb(0x7b6df0));
                const uint8_t ek = lk.kind == "cross" ? E_CROSS : E_SELF;

                for (int p = 0; p < dst_cnt; ++p) {
                    int r = p - lk.row_offset;
                    if (r < 0 || r >= R) continue;
                    float c_imp = imp[static_cast<size_t>(dst)][static_cast<size_t>(p)];
                    const float* row = &mat[static_cast<size_t>(r) * static_cast<size_t>(C)];
                    int lim = lk.causal ? std::min(C, p + 1) : std::min(C, src_cnt);
                    if (c_imp > 0.0f) {
                        for (int q = 0; q < lim; ++q)
                            imp[static_cast<size_t>(src)][static_cast<size_t>(q)] += per_attn * c_imp * row[q];
                    }
                    top_k_indices(row, lim, K, 0.005f, idx, pick);
                    for (int q : pick)
                        push(n.node(src, q), n.node(dst, p), row[q], ek, dst, per_attn * c_imp * row[q], col);
                }
            }
        }
    }

    for (int s = 0; s < ns; ++s) {
        float mx = 1e-12f;
        for (float x : imp[static_cast<size_t>(s)]) mx = std::max(mx, x);
        for (int i = 0; i < n.count[static_cast<size_t>(s)]; ++i)
            n.nodes[static_cast<size_t>(n.node(s, i))].imp = imp[static_cast<size_t>(s)][static_cast<size_t>(i)] / mx;
    }
    for (int o = 0; o < n.count[static_cast<size_t>(n.s_out)]; ++o)
        n.nodes[static_cast<size_t>(n.out_first + o)].imp = n.nodes[static_cast<size_t>(n.out_first + o)].act;

    std::vector<float> stage_max(static_cast<size_t>(n.nstages), 1e-12f);
    for (size_t e = 0; e < n.edges.size(); ++e)
        if (n.edges[e].kind != E_RESID) stage_max[static_cast<size_t>(n.edges[e].stage)] = std::max(stage_max[static_cast<size_t>(n.edges[e].stage)], edge_flow[e]);
    for (size_t e = 0; e < n.edges.size(); ++e)
        n.edges[e].imp = std::min(1.0f, edge_flow[e] / stage_max[static_cast<size_t>(n.edges[e].stage)]);

    std::vector<int> order(n.edges.size());
    std::iota(order.begin(), order.end(), 0);
    size_t keep = std::min<size_t>(900, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep), order.end(),
                      [&](int p, int q) { return n.edges[static_cast<size_t>(p)].imp > n.edges[static_cast<size_t>(q)].imp; });
    for (size_t i = 0; i < keep; ++i)
        if (n.edges[static_cast<size_t>(order[i])].imp >= 0.2f) n.hot.push_back(order[i]);
    return net;
}

void project(Net& n, const Camera3D& c) {
    for (NNode& nd : n.nodes) c.map(nd.x, nd.y, nd.z, &nd.sx, &nd.sy, &nd.scale);
}

int pick_node(const Net& n, int x, int y) {
    int best = -1;
    float best_d = 1e9f;
    for (size_t i = 0; i < n.nodes.size(); ++i) {
        float dx = n.nodes[i].sx - static_cast<float>(x), dy = n.nodes[i].sy - static_cast<float>(y);
        float dd = dx * dx + dy * dy;
        float reach = std::max(5.0f, 3.5f * n.nodes[i].scale);
        if (dd < reach * reach && dd < best_d) { best_d = dd; best = static_cast<int>(i); }
    }
    return best;
}

std::string node_text(const VitroApp& a, const Net& n, int i) {
    const UnitTrace& u = a.U();
    const NNode& nd = n.nodes[static_cast<size_t>(i)];
    if (nd.stage == n.s_out) {
        return fmt("output \"%s\" (p=%.3f)%s", a.d.label(nd.idx).c_str(), nd.act, i == n.true_out ? " [target]" : "");
    }
    const StageDef& st = u.stages[static_cast<size_t>(nd.stage)];
    std::string s = fmt("%s [%d]", st.label.c_str(), nd.idx);
    if (st.token_source == "target") s += fmt(" \"%s\"", a.token_text(u, nd.idx + st.token_offset).c_str());
    else if (st.token_source == "source") s += fmt(" \"%s\"", a.source_text(u, nd.idx + st.token_offset).c_str());
    s += fmt(" | act %.2f, rollout %.2f", nd.act, nd.imp);
    return s;
}

void side_panel(VitroApp& a, Rect r, const Net& n) {
    Canvas& cv = a.cv;
    const TraceDump& d = a.d;
    const UnitTrace& u = a.U();
    a.panel(r, "3D Trace & Attention Rollout");
    int x = r.x + cv.s(12), w = r.w - cv.s(24), y = r.y + cv.s(30);
    int j = n.token;
    cv.text_fit(x, y, w, fmt("token %d  \"%s\"", j, a.token_text(u, j).c_str()), pal::text, F_BIG);
    y += cv.s(30);
    if (j >= u.t0 && j < u.n_tokens()) {
        size_t jj = static_cast<size_t>(j);
        cv.text_fit(x, y, w, fmt("p=%.4f  |  rank %d", u.p.f(jj), u.rank.iv(jj)), u.rank.iv(jj) > 1 ? pal::bad : pal::text, F_UI);
    }
    y += cv.s(20);
    cv.text_fit(x, y, w, u.label, pal::dim, F_SMALL);
    y += cv.s(26);
    cv.text(x, y, "Top Predictions", pal::dim, F_SMALL);
    y += cv.s(18);
    for (int o = 0; o < n.count[static_cast<size_t>(n.s_out)]; ++o) {
        const NNode& nd = n.nodes[static_cast<size_t>(n.out_first + o)];
        bool right = n.out_first + o == n.true_out;
        cv.text_fit(x, y, cv.s(66), d.label(nd.idx), right ? pal::good : pal::text, F_MONO, 2);
        a.bar({x + cv.s(72), y + cv.s(3), w - cv.s(134), cv.s(11)}, nd.act, right ? pal::good : pal::accent);
        cv.text(x + w - cv.s(54), y, fmt("%.3f", nd.act), pal::dim, F_SMALL);
        y += cv.s(19);
    }
    y += cv.s(14);
    int bw = (w - cv.s(8)) / 2;
    a.button({x, y, bw, cv.s(26)}, a.st.net_play ? "Pause Wave" : "Play Wave", a.st.net_play, H_NET_BTN, 0, 0, F_SMALL);
    a.button({x + bw + cv.s(8), y, bw, cv.s(26)}, a.st.net_path ? "Rollout Only" : "All Top-K Edges", a.st.net_path, H_NET_BTN, 1, 0, F_SMALL);
    y += cv.s(34);
    cv.text(x, y, "Incoming edges per node (K)", pal::dim, F_SMALL);
    y += cv.s(18);
    static const int ks[] = {1, 2, 3, 4, 6, 8};
    int kw = (w - cv.s(5) * 5) / 6;
    for (int i = 0; i < 6; ++i) a.button({x + i * (kw + cv.s(5)), y, kw, cv.s(24)}, fmt("%d", ks[i]), a.st.net_k == ks[i], H_NET_BTN, 10 + ks[i], 0, F_SMALL);
    y += cv.s(32);
    a.button({x, y, w, cv.s(24)}, "Reset Camera (R)", false, H_NET_BTN, 2, 0, F_SMALL);
    y += cv.s(38);
    cv.fill({x, y + cv.s(4), cv.s(24), cv.s(3)}, rgb(0xffc94a));
    cv.text_fit(x + cv.s(32), y, w - cv.s(32), "Gold: backward rollout influence path", pal::text, F_SMALL);
    y += cv.s(20);
    cv.fill({x, y + cv.s(4), cv.s(24), cv.s(2)}, rgb(0x7b6df0));
    cv.text_fit(x + cv.s(32), y, w - cv.s(32), "Violet / Cyan: strongest attention links", pal::text, F_SMALL);
    y += cv.s(24);
    cv.text_fit(x, y, w, fmt("stages %d | nodes %d | edges %d", n.nstages, static_cast<int>(n.nodes.size()), static_cast<int>(n.edges.size())), pal::faint, F_SMALL);
}

}  // namespace

void draw_net(VitroApp& a, Rect area) {
    Canvas& cv = a.cv;
    const UnitTrace& u = a.U();
    const int j_sel = selected_token(a, u);
    std::shared_ptr<Net> net = std::static_pointer_cast<Net>(a.net);
    if (!net || net->unit != a.st.unit || net->token != j_sel || net->head != a.st.head || net->k != a.st.net_k) {
        net = build_net(a, u, j_sel);
        a.net = net;
    }
    Net& n = *net;
    const int pad = cv.s(10), panel_w = cv.s(330);
    Rect view = {area.x + pad, area.y + pad, area.w - panel_w - 3 * pad, area.h - 2 * pad};
    cv.fill(view, rgb(0x0b0a1a));
    cv.frame(view, pal::border);
    side_panel(a, {view.right() + pad, view.y, panel_w, view.h}, n);

    const bool still_camera = n.cam_ok && n.cam_yaw == a.st.net_yaw && n.cam_pitch == a.st.net_pitch && n.cam_zoom == a.st.net_zoom &&
                              n.cam_view.x == view.x && n.cam_view.y == view.y && n.cam_view.w == view.w && n.cam_view.h == view.h;
    Camera3D c;
    if (still_camera) {
        c = n.cam;
    } else {
        c.cy = std::cos(a.st.net_yaw); c.sy = std::sin(a.st.net_yaw);
        c.cp = std::cos(a.st.net_pitch); c.sp = std::sin(a.st.net_pitch);
        c.half_span = n.half_span;
        c.dist = std::max(120.0f, n.half_span * 3.4f);
        c.vx = static_cast<float>(view.x + view.w / 2);
        c.vy = static_cast<float>(view.y + view.h / 2 + cv.s(8));
        c.ox = c.vx; c.oy = c.vy; c.focal = c.dist;
        if (!n.box_ok || n.box_yaw != a.st.net_yaw || n.box_pitch != a.st.net_pitch) {
            project(n, c);
            float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
            for (const NNode& nd : n.nodes) {
                minx = std::min(minx, nd.sx); maxx = std::max(maxx, nd.sx);
                miny = std::min(miny, nd.sy); maxy = std::max(maxy, nd.sy);
            }
            n.box_x0 = minx - c.vx; n.box_x1 = maxx - c.vx; n.box_y0 = miny - c.vy; n.box_y1 = maxy - c.vy;
            n.box_yaw = a.st.net_yaw; n.box_pitch = a.st.net_pitch; n.box_ok = true;
        }
        float fit = std::min(0.92f * static_cast<float>(view.w) / std::max(1.0f, n.box_x1 - n.box_x0),
                             0.74f * static_cast<float>(view.h) / std::max(1.0f, n.box_y1 - n.box_y0));
        c.focal *= fit;
        c.ox = c.vx - (n.box_x0 + n.box_x1) / 2 * fit;
        c.oy = c.vy - (n.box_y0 + n.box_y1) / 2 * fit;
        c.zoom = a.st.net_zoom;
        project(n, c);
        n.cam = c; n.cam_yaw = a.st.net_yaw; n.cam_pitch = a.st.net_pitch; n.cam_zoom = a.st.net_zoom; n.cam_view = view;
        n.cam_ok = true;
    }

    const auto frame_start = std::chrono::steady_clock::now();
    cv.push_clip(view);
    const bool still = a.dragging && a.moved;
    if (a.adaptive) {
        if (n.last_ms > 12.5) n.lod = std::max(0.1f, n.lod * 0.85f);
        else if (n.last_ms < 7.0) n.lod = std::min(1.0f, n.lod * 1.1f);
    }
    const float lod = std::max(0.08f, a.adaptive && still ? n.lod * 0.5f : (a.adaptive ? n.lod : 1.0f));
    const float faint_gain = std::min(6.0f, 1.0f / lod);
    const bool wave = a.st.net_play || a.st.net_time > 0.0;
    const double wave_pos = wave ? std::fmod(a.st.net_time / WAVE_PERIOD, 1.0) * static_cast<double>(std::max(1, n.nstages - 1)) : -10.0;
    const Color gold = rgb(0xffc94a), violet = rgb(0x7b6df0);

    for (size_t ei = 0; ei < n.edges.size(); ++ei) {
        const NEdge& e = n.edges[ei];
        if (a.st.net_path && e.imp < 0.10f) continue;
        const bool faint = e.imp < 0.10f || (e.kind == E_RESID && e.imp < 0.12f);
        if (faint && lod < 1.0f && hash01(static_cast<uint32_t>(ei), 5u) >= lod) continue;
        const float gain = faint ? faint_gain : 1.0f;
        const NNode& p = n.nodes[static_cast<size_t>(e.a)];
        const NNode& q = n.nodes[static_cast<size_t>(e.b)];
        float t = e.imp;
        float boost = wave ? 1.0f + 1.7f * static_cast<float>(std::exp(-std::pow(static_cast<double>(e.stage) - 0.5 - wave_pos, 2.0) / 1.4)) : 1.0f;
        if (e.kind == E_RESID && t < 0.12f) {
            cv.line_add(p.sx, p.sy, q.sx, q.sy, violet, std::min(1.0f, 0.022f * boost * gain));
        } else if (t < 0.10f) {
            cv.line_add(p.sx, p.sy, q.sx, q.sy, e.tint, std::min(1.0f, (0.025f + 0.11f * std::sqrt(std::min(1.0f, e.w))) * boost * gain));
        } else {
            float g = std::pow(t, 0.9f);
            Color col = mix(e.tint, gold, std::min(1.0f, g * 1.5f));
            float al = (0.08f + 0.62f * g) * boost * (e.kind == E_RESID ? 0.55f : 1.0f);
            cv.line_add(p.sx, p.sy, q.sx, q.sy, col, std::min(1.0f, al));
            if (g > 0.7f) cv.line_add(p.sx + 1, p.sy, q.sx + 1, q.sy, col, std::min(1.0f, al * 0.6f));
        }
    }

    for (size_t i = 0; i < n.nodes.size(); ++i) {
        const NNode& nd = n.nodes[i];
        bool is_lat = nd.stage < static_cast<int>(u.stages.size()) && u.stages[static_cast<size_t>(nd.stage)].kind == "latent";
        float g = nd.stage == n.s_out ? 1.0f : std::pow(nd.imp, 0.6f);
        Color col = nd.stage == n.s_out ? (static_cast<int>(i) == n.true_out ? rgb(0x6ee7a0) : rgb(0xff8a8a)) : mix(rgb(0x9a90ff), gold, g);
        float crowd = std::clamp(70.0f / static_cast<float>(std::max(1, n.count[static_cast<size_t>(nd.stage)])), 0.22f, 1.0f);
        float r = (is_lat ? 3.0f : (0.9f + 1.5f * g) * std::sqrt(crowd) + 0.3f) * std::min(1.8f, std::max(0.7f, nd.scale * 1.1f));
        float al = nd.stage == n.s_out ? 0.9f : std::min(1.0f, (0.10f + 0.35f * nd.act + 0.7f * g) * (is_lat ? 1.4f : std::sqrt(crowd) + 0.15f));
        if (still && g < 0.2f && (i & 1)) continue;
        cv.glow(nd.sx, nd.sy, r * 2.0f, col, al * 0.7f);
        cv.add_px(static_cast<int>(nd.sx), static_cast<int>(nd.sy), col, std::min(1.0f, al + 0.25f));
    }
    n.last_ms = 0.7 * n.last_ms + 0.3 * std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frame_start).count();

    if (n.s_pred < n.s_out && n.count[static_cast<size_t>(n.s_pred)] > 0) {
        const NNode& s = n.nodes[static_cast<size_t>(n.node(n.s_pred, n.sel_p))];
        cv.glow(s.sx, s.sy, 11.0f * std::max(0.8f, s.scale), rgb(0xffe08a), 0.9f);
        for (int o = 0; o < n.count[static_cast<size_t>(n.s_out)]; ++o) {
            const NNode& nd = n.nodes[static_cast<size_t>(n.out_first + o)];
            cv.glow(nd.sx, nd.sy, 8.0f * std::max(0.8f, nd.scale), n.out_first + o == n.true_out ? rgb(0x6ee7a0) : rgb(0xff8a8a), 0.8f);
            cv.text(static_cast<int>(nd.sx) + cv.s(10), static_cast<int>(nd.sy) - cv.s(7), a.d.label(nd.idx), n.out_first + o == n.true_out ? pal::good : pal::text, F_SMALL);
        }
    }

    if (wave) {
        for (int ei : n.hot) {
            const NEdge& e = n.edges[static_cast<size_t>(ei)];
            const NNode& p = n.nodes[static_cast<size_t>(e.a)];
            const NNode& q = n.nodes[static_cast<size_t>(e.b)];
            float ph = std::fmod(static_cast<float>(a.st.net_time) * 0.42f + hash01(static_cast<uint32_t>(ei), 77u), 1.0f);
            for (int tail = 0; tail < 4; ++tail) {
                float tt = ph - 0.025f * static_cast<float>(tail);
                if (tt < 0.0f) tt += 1.0f;
                cv.glow(p.sx + (q.sx - p.sx) * tt, p.sy + (q.sy - p.sy) * tt, 3.2f - 0.5f * static_cast<float>(tail), rgb(0xfff2c0),
                        (0.85f - 0.2f * static_cast<float>(tail)) * std::min(1.0f, 0.4f + e.imp));
            }
        }
    }

    StaggeredLabels stagger;
    for (const StageGroupSpan& g : n.groups) {
        float x0, y0, x1, y1, sc0, sc1;
        c.map(n.stage_x[static_cast<size_t>(g.a)], -162.0f, 0.0f, &x0, &y0, &sc0);
        c.map(n.stage_x[static_cast<size_t>(g.b)], -162.0f, 0.0f, &x1, &y1, &sc1);
        bool lit = wave && wave_pos + 0.5 >= g.a && wave_pos - 0.5 <= g.b;
        Color col = lit ? pal::select : pal::dim;
        cv.line(static_cast<int>(x0), static_cast<int>(y0), static_cast<int>(x1), static_cast<int>(y1), col);
        int tw = cv.text_width(g.label, F_SMALL), cx = static_cast<int>((x0 + x1) / 2);
        int row = stagger.place(cx, tw, cv.s(8));
        cv.text(cx - tw / 2, static_cast<int>((y0 + y1) / 2) + cv.s(4) + row * (cv.fonts.height(F_SMALL) + cv.s(2)), g.label, col, F_SMALL);
    }
    cv.pop_clip();
    cv.text_fit(view.x + cv.s(12), view.y + cv.s(8), view.w - cv.s(24), "3D Graph: drag to orbit, scroll wheel to zoom, click any node to inspect token", pal::dim, F_SMALL);
    a.add_hit(view, H_NET_VIEW, 0, 0);
}

void net_drag(VitroApp& a, int dx, int dy) {
    a.st.net_yaw += static_cast<float>(dx) * 0.008f;
    a.st.net_pitch = std::clamp(a.st.net_pitch + static_cast<float>(dy) * 0.008f, -1.45f, 1.45f);
}

void net_pick_click(VitroApp& a, int x, int y) {
    std::shared_ptr<Net> n = std::static_pointer_cast<Net>(a.net);
    if (!n) return;
    int i = pick_node(*n, x, y);
    if (i < 0) return;
    const NNode& nd = n->nodes[static_cast<size_t>(i)];
    if (nd.stage >= static_cast<int>(a.U().stages.size())) return;
    const StageDef& st = a.U().stages[static_cast<size_t>(nd.stage)];
    if (st.token_source == "target") {
        int tok = nd.idx + st.token_offset;
        if (tok >= a.U().t0 && tok < a.U().n_tokens()) a.select_token(tok);
    }
}

bool describe_net(VitroApp& a, const Hit& h) {
    if (h.kind == H_NET_BTN) {
        a.status = h.a == 0 ? "Play / pause activation wave (Space)" : h.a == 1 ? "Toggle rollout-only path vs all edges (P)" : h.a == 2 ? "Reset 3D camera (R)" : "Top-K incoming links per node (K)";
        return true;
    }
    if (h.kind != H_NET_VIEW) return false;
    std::shared_ptr<Net> n = std::static_pointer_cast<Net>(a.net);
    if (!n) return true;
    int i = pick_node(*n, a.mx, a.my);
    a.status = i >= 0 ? node_text(a, *n, i) : "Drag to orbit 3D view, scroll wheel to zoom, hover nodes to inspect";
    return true;
}

}  // namespace vitro
