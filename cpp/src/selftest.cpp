// Self-test runner for vitro_view (--selftest).
#include <cstdio>

#include <vitro/selftest_core.hpp>
#include "view_app.hpp"

namespace vitro {

int run_selftest() {
    SelftestResult r;
    run_core_selftests(r);
    std::printf("selftest: %d checks, %d failed\n", r.checks, r.failed);
    return r.failed ? 2 : 0;
}

}  // namespace vitro
