// Start screen and trace library scanner (.vtdump).
#include <algorithm>
#include <ctime>

#include "view_app.hpp"

namespace vitro {

void draw_start_header(VitroApp& a) {
    Canvas& cv = a.cv;
    Rect hd = {0, 0, cv.W, a.header_h()};
    cv.fill(hd, pal::panel);
    cv.hline(0, cv.W - 1, hd.bottom() - 1, pal::border);
    cv.text(cv.s(12), cv.s(6), "Vitro", pal::text, F_BIG);
    cv.text_fit(cv.s(12), cv.s(36), cv.W - cv.s(24), "Glass-box PyTorch model trace inspector: tokens, attention, activations, user spans, and 3D network flow.", pal::dim, F_SMALL);
}

void draw_start_status(VitroApp& a) {
    Canvas& cv = a.cv;
    Rect sr = {0, cv.H - a.status_h(), cv.W, a.status_h()};
    cv.fill(sr, pal::panel);
    cv.hline(0, cv.W - 1, sr.y, pal::border);
    std::string msg = a.status.empty() ? "Ctrl+O open trace | Drag & drop .vtdump file | F1 help" : a.status;
    cv.text_fit(cv.s(10), sr.y + cv.s(4), cv.W - cv.s(220), msg, pal::text, F_SMALL);
    cv.text(0, sr.y + cv.s(4), fmt("traces found: %d", static_cast<int>(a.library_items.size())), pal::dim, F_SMALL, cv.W - cv.s(10), 2);
}

void draw_library(VitroApp& a, Rect r) {
    Canvas& cv = a.cv;
    if (a.library_dirty) a.scan_library();
    cv.fill(r, pal::bg);
    int x = r.x + cv.s(40), y = r.y + cv.s(26), w = std::min(r.w - cv.s(80), cv.s(1000));
    cv.text(x, y, "Trace Library", pal::text, F_BIG);
    y += cv.s(34);
    cv.text_fit(x, y, w, "Select a trace below, drag & drop a .vtdump file into this window, or use File -> Open Trace (Ctrl+O).", pal::dim, F_UI);
    y += cv.s(24);
    cv.text_fit(x, y, w, "Folder: " + (a.library_dir.empty() ? "(none)" : a.library_dir), pal::faint, F_SMALL);
    y += cv.s(26);
    if (!a.error.empty()) {
        cv.text_fit(x, y, w, a.error, pal::bad, F_UI);
        y += cv.s(26);
    }
    a.button({x, y, cv.s(180), cv.s(32)}, "Open File...", false, H_LIB_OPEN, 0, 0, F_UI);
    if (a.d.loaded()) a.button({x + cv.s(192), y, cv.s(220), cv.s(32)}, "Return to Active Trace", false, H_LIB_OPEN, 1, 0, F_UI);
    y += cv.s(50);
    if (a.library_items.empty()) {
        cv.text_fit(x, y, w, "No .vtdump files found yet. Generate one with vitro.export_dump(...) in Python.", pal::dim, F_UI);
        return;
    }
    cv.text(x, y, "Recent & Folder Traces", pal::dim, F_SMALL);
    y += cv.s(22);
    const int row_h = cv.s(52);
    Rect list = {r.x, y, r.w, r.bottom() - y};
    a.main_sv = {list, static_cast<int>(a.library_items.size()) * row_h};
    a.library_scroll = std::clamp(a.library_scroll, 0, a.main_sv.max_offset());
    cv.push_clip(list);
    for (size_t i = 0; i < a.library_items.size(); ++i) {
        const LibraryItem& it = a.library_items[i];
        Rect row = {x, y - a.library_scroll + static_cast<int>(i) * row_h, w, cv.s(46)};
        if (row.bottom() < list.y || row.y > list.bottom()) continue;
        Rect shown = row.intersect(list);
        bool hov = shown.contains(a.mx, a.my);
        cv.fill(row, hov ? rgb(0x2a3f5e) : pal::panel);
        cv.frame(row, hov ? pal::accent : pal::border);
        cv.text_fit(row.x + cv.s(14), row.y + cv.s(5), row.w - cv.s(240), it.name, pal::text, F_UI);
        cv.text_fit(row.x + cv.s(14), row.y + cv.s(25), row.w - cv.s(240), it.path, pal::faint, F_SMALL);
        std::time_t tt = static_cast<std::time_t>(it.mtime);
        char when[40];
        std::strftime(when, sizeof when, "%Y-%m-%d %H:%M", std::localtime(&tt));
        cv.text(row.x, row.y + cv.s(6), fmt("%.2f MB", static_cast<double>(it.bytes) / (1 << 20)), pal::dim, F_SMALL, row.w - cv.s(14), 2);
        cv.text(row.x, row.y + cv.s(25), when, pal::dim, F_SMALL, row.w - cv.s(14), 2);
        a.add_hit(shown, H_LIB_ITEM, static_cast<int>(i));
    }
    cv.pop_clip();
}

void describe_library(VitroApp& a, const Hit& h) {
    switch (h.kind) {
        case H_LIB_ITEM: a.status = "open trace: " + a.library_items[static_cast<size_t>(h.a)].path; break;
        case H_LIB_OPEN: a.status = h.a == 0 ? "browse for .vtdump file (Ctrl+O)" : "return to loaded trace (B)"; break;
        case H_SCROLL: a.status = "scrollbar: click or drag, mouse wheel, PgUp/PgDn/Home/End"; break;
        default: break;
    }
}

void describe_hit(VitroApp& a, const Hit& h) {
    if (h.kind == H_LIB_ITEM || h.kind == H_LIB_OPEN || h.kind == H_SCROLL) {
        describe_library(a, h);
        return;
    }
    const UnitTrace& u = a.U();
    switch (h.kind) {
        case H_TOKEN: a.status = a.describe_token(u, h.a); break;
        case H_FREE_TOKEN: {
            int j = h.a;
            if (j >= u.free_n()) { a.status = fmt("token %d: rollout ended at step %d", j, u.free_n()); break; }
            bool same = u.ftok.iv(static_cast<size_t>(j)) == u.tok.iv(static_cast<size_t>(j));
            a.status = fmt("token %d: free generated \"%s\" (p=%.3f) - %s \"%s\"", j, a.free_text(u, j).c_str(), u.fp.f(static_cast<size_t>(j)),
                           same ? "matches" : "differs from", a.token_text(u, j).c_str());
            break;
        }
        case H_SIDE_UNIT: {
            const UnitTrace& su = a.d.units[static_cast<size_t>(h.a)];
            a.status = fmt("%s (%d tokens): forced %s, bits %.1f", su.label.c_str(), su.n_tokens(), su.forced_exact ? "exact" : "has errors", su.bits);
            break;
        }
        default:
            if (describe_attention(a, h) || describe_activations(a, h) || describe_spans(a, h) || describe_net(a, h)) return;
            break;
    }
}

}  // namespace vitro
