// View 4, "Spans": user-annotated prompt/sequence regions and hierarchical sub-parts.
#include <algorithm>
#include <cmath>

#include "view_app.hpp"

namespace vitro {

namespace {

int span_depth(const TraceDump& d, int idx) {
    int depth = 0, cur = idx;
    for (int guard = 0; guard < 32 && cur >= 0 && cur < static_cast<int>(d.spans.size()); ++guard) {
        int p = d.spans[static_cast<size_t>(cur)].parent;
        if (p < 0 || p == cur) break;
        ++depth;
        cur = p;
    }
    return depth;
}

}  // namespace

void draw_spans(VitroApp& a, Rect area) {
    Canvas& cv = a.cv;
    const TraceDump& d = a.d;
    const UnitTrace& u = a.U();
    Rect r = area.inset(cv.s(10));
    a.panel(r, "");
    cv.text(r.x + cv.s(12), r.y + cv.s(8),
            fmt("Annotated Sequence Spans / Parts (%d spans across %d tokens) - click any span to jump",
                static_cast<int>(d.spans.size()), u.n_tokens()),
            pal::text, F_BOLD);

    if (d.spans.empty()) {
        cv.text_fit(r.x + cv.s(16), r.y + cv.s(46), r.w - cv.s(32),
                    "No custom spans were attached. Pass SpanLabel(...) in Python TraceSpec to organize prompts, AST blocks, or tree parts.",
                    pal::dim, F_UI);
        return;
    }

    int pad_y = cv.s(38), row_h = cv.s(46);
    Rect list = {r.x + cv.s(8), r.y + pad_y, r.w - cv.s(16), r.h - pad_y - cv.s(8)};
    a.main_sv = {list, static_cast<int>(d.spans.size()) * row_h + cv.s(8)};
    a.st.scroll = std::clamp(a.st.scroll, 0, a.main_sv.max_offset());
    cv.push_clip(list);

    int total_tok = std::max(1, u.n_tokens());
    int y = list.y - a.st.scroll;
    for (size_t i = 0; i < d.spans.size(); ++i) {
        const SpanInfo& sp = d.spans[i];
        Rect card = {list.x + cv.s(4), y, list.w - cv.s(18), row_h - cv.s(6)};
        y += row_h;
        if (card.bottom() < list.y || card.y > list.bottom()) continue;

        int depth = span_depth(d, static_cast<int>(i));
        int lo = std::clamp(sp.start, 0, u.n_tokens()), hi = std::clamp(sp.end, lo, u.n_tokens());
        int errors = 0;
        double bits = 0.0;
        for (int j = std::max(lo, u.t0); j < hi; ++j) {
            size_t jj = static_cast<size_t>(j);
            bits += -std::log2(std::max(u.p.f(jj), 1e-30f));
            if (u.rank.iv(jj) > 1) ++errors;
        }
        Color tint = sp.tint.empty() ? pal::accent : parse_tint(sp.tint, pal::accent);
        bool sel = static_cast<int>(i) == a.st.span_sel;
        cv.fill(card, sel ? rgb(0x273549) : pal::panel2);
        cv.frame(card, sel ? pal::select : pal::border);
        cv.fill({card.x, card.y, cv.s(5), card.h}, tint);

        int text_x = card.x + cv.s(14 + depth * 18);
        cv.text_fit(text_x, card.y + cv.s(4), card.w - cv.s(420), fmt("[%s] %s", sp.kind.c_str(), sp.label.c_str()), pal::text, F_BOLD);
        cv.text_fit(text_x, card.y + cv.s(21), card.w - cv.s(420),
                    sp.meta.empty() ? fmt("tokens %d..%d (%d tok)", sp.start, sp.end - 1, std::max(0, hi - lo)) : sp.meta,
                    pal::dim, F_SMALL);

        Rect track = {card.right() - cv.s(360), card.y + cv.s(8), cv.s(210), cv.s(12)};
        cv.fill(track, pal::panel);
        int bx = track.x + static_cast<int>(static_cast<long long>(lo) * track.w / total_tok);
        int bw = std::max(3, static_cast<int>(static_cast<long long>(hi - lo) * track.w / total_tok));
        cv.fill({bx, track.y, bw, track.h}, tint);
        cv.frame(track, pal::border);

        cv.text(card.right() - cv.s(138), card.y + cv.s(8), fmt("%d err | %.1f b", errors, bits), errors ? pal::warn : pal::good, F_SMALL, cv.s(128), 2);
        a.add_hit(card.intersect(list), H_SPAN_ROW, static_cast<int>(i), lo);
    }
    cv.pop_clip();
}

bool describe_spans(VitroApp& a, const Hit& h) {
    if (h.kind != H_SPAN_ROW) return false;
    const SpanInfo& sp = a.d.spans[static_cast<size_t>(h.a)];
    a.status = fmt("span \"%s\" (%s): tokens %d..%d - click to open in Tokens", sp.label.c_str(), sp.kind.c_str(), sp.start, sp.end - 1);
    return true;
}

}  // namespace vitro
