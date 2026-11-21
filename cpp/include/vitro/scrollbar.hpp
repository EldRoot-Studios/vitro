// Vertical scrollbar geometry and drag arithmetic shared across Vitro views.
#pragma once
#include <algorithm>

#include "canvas.hpp"
#include "colors.hpp"

namespace vitro {

struct ScrollView {
    Rect area;
    int content = 0;
    bool needed() const { return area.h > 0 && content > area.h; }
    int max_offset() const { return std::max(0, content - area.h); }
};

inline Rect scrollbar_track(const Canvas& cv, const ScrollView& sv) {
    int w = cv.s(9);
    return {sv.area.right() - w - cv.s(2), sv.area.y + cv.s(2), w, sv.area.h - cv.s(4)};
}

inline Rect scrollbar_thumb(const Canvas& cv, const ScrollView& sv, int offset) {
    Rect t = scrollbar_track(cv, sv);
    int th = std::min(t.h, std::max(cv.s(26), static_cast<int>(static_cast<long long>(t.h) * sv.area.h / std::max(1, sv.content))));
    int span = t.h - th, mo = sv.max_offset();
    int ty = t.y + (mo > 0 ? static_cast<int>(static_cast<long long>(span) * std::clamp(offset, 0, mo) / mo) : 0);
    return {t.x, ty, t.w, th};
}

inline int scrollbar_offset_at(const Canvas& cv, const ScrollView& sv, int y) {
    Rect t = scrollbar_track(cv, sv), th = scrollbar_thumb(cv, sv, 0);
    int span = t.h - th.h;
    if (span <= 0) return 0;
    long long pos = std::clamp(y - th.h / 2 - t.y, 0, span);
    return static_cast<int>(pos * sv.max_offset() / span);
}

inline Rect draw_scrollbar(Canvas& cv, const ScrollView& sv, int offset, bool hot) {
    if (!sv.needed()) return Rect{};
    Rect t = scrollbar_track(cv, sv), th = scrollbar_thumb(cv, sv, offset);
    cv.fill(t, pal::panel2, 0.55f);
    cv.fill(th, hot ? pal::accent : rgb(0x56606c));
    return t;
}

}  // namespace vitro
