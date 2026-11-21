// Reads a Vitro recorded generation (.vtdemo, magic "VITRO01\n", kind "demo") with dynamic stages and links.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <vitro/container.hpp>
#include "vitro_file.hpp"

namespace vitro {

inline constexpr double DEMO_READ_TICKS = 8.0;

struct PrefillLink {
    int src = 0, dst = 0;
    std::string kind, tint;
    Arr ids, weights;           // (C_dst, K) int32 & float
};

struct DemoTape {
    Arr top_id, top_p;          // (M, top_k)
    Arr entropy, p_ref;         // (M,), optional (M,)
    Arr self_id, self_w;        // (M, Ld, K_self)
    Arr cross_id, cross_w;      // optional (M, Ld, K_cross)
    Arr norm;                   // (M, Ld, C_norm)
};

struct DemoSegment {
    int k = 0;
    std::string label, kind;
    int t0 = 1, diverge = -1;
    bool free_exact = false, free_ended = false, has_read = false;
    double read_ticks = 0.0, write_ticks = 0.0, start = 0.0;
    double end() const { return start + read_ticks + write_ticks; }

    Arr src_tok, ref_tok;       // optional reference/source token streams
    Arr ftok, fp;               // written tokens (M,) & probabilities (M,)
    std::vector<StageDef> stages;
    std::vector<PrefillLink> prefill_links;

    int in_stage = 0, out_stage = 0, cross_stage = -1;
    std::vector<int> layer_stages;
    DemoTape tape;

    int free_n() const { return static_cast<int>(ftok.n); }
};

class TraceDemo {
public:
    TraceDemo() = default;
    explicit TraceDemo(const std::string& p) { open(p); }
    TraceDemo(TraceDemo&&) = default;
    TraceDemo& operator=(TraceDemo&&) = default;
    TraceDemo(const TraceDemo&) = delete;
    TraceDemo& operator=(const TraceDemo&) = delete;

    bool loaded() const { return !segments.empty(); }

    std::string path, uid, title, model_name, arch, precision, created;
    Json head;
    std::vector<std::string> labels;
    std::vector<SpanInfo> spans;
    std::vector<DemoSegment> segments;
    long long step = -1;
    int top_k = 5, total_tokens = 0;
    double total = 0.0;
    size_t data_size = 0;

    const std::string& label(int id) const {
        if (id < 0 || id >= static_cast<int>(labels.size()))
            throw std::runtime_error("token id " + std::to_string(id) + " outside vocabulary");
        return labels[static_cast<size_t>(id)];
    }

    int segment_at(double pos) const {
        int lo = 0, hi = static_cast<int>(segments.size()) - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (segments[static_cast<size_t>(mid)].start <= pos) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }

private:
    MappedContainer map_;

    void open(const std::string& p) {
        path = p;
        map_.open(p, "VITRO01\n", "not a Vitro trace file (expected VITRO01 magic)");
        head = map_.head;
        data_size = map_.data_size;
        const std::string kind = head.str_or("kind", "");
        if (kind != "demo")
            throw std::runtime_error(p + ": this is a '" + kind + "' trace, not a generation demo (open with vitro_view)");

        const Json& tr = head["trace"];
        uid = tr.str_or("id", "demo");
        title = tr.str_or("title", uid);
        model_name = tr.str_or("model_name", "PyTorch Model");
        arch = tr.str_or("arch", "decoder_only");
        precision = tr.str_or("precision", "fp32");
        created = head.str_or("created", "");
        step = tr.has("step") && !tr["step"].is_null() ? static_cast<long long>(tr["step"].num()) : -1;
        top_k = tr.i_or("top_k", 5);
        labels = json_strings(head["vocab"]["labels"]);

        if (head.has("spans") && !head["spans"].is_null()) {
            const Json& sp = head["spans"];
            for (size_t i = 0; i < sp.size(); ++i) {
                const Json& js = sp.at(i);
                spans.push_back({js.i_or("id", static_cast<int>(i)), js["start"].i(), js["end"].i(),
                                 js.i_or("parent", -1), js["label"].str(), js.str_or("kind", "region"),
                                 js.str_or("tint", ""), js.str_or("meta", ""), {}});
            }
        }

        const Json& segs = head["segments"];
        if (segs.size() == 0) throw std::runtime_error(p + ": demo contains zero segments");
        double t = 0.0;
        for (size_t si = 0; si < segs.size(); ++si) {
            const Json& j = segs.at(si);
            DemoSegment S;
            S.k = j.i_or("k", static_cast<int>(si));
            S.label = j.str_or("label", "Segment " + std::to_string(S.k));
            S.kind = j.str_or("kind", "decode");
            S.t0 = std::max(1, j.i_or("t0", 1));
            S.diverge = j.i_or("diverge", -1);
            S.free_exact = j.bool_or("free_exact", false);
            S.free_ended = j.bool_or("free_ended", false);
            S.has_read = j.bool_or("has_read", false);
            S.read_ticks = S.has_read ? j.num_or("read_ticks", DEMO_READ_TICKS) : 0.0;

            if (j.has("source_tokens") && !j["source_tokens"].is_null()) S.src_tok = map_.arr(j["source_tokens"]);
            if (j.has("ref_tokens") && !j["ref_tokens"].is_null()) S.ref_tok = map_.arr(j["ref_tokens"]);
            S.ftok = map_.arr(j["free_tok"]);
            S.fp = map_.arr(j["free_p"]);
            total_tokens += S.free_n();
            if (S.free_n() <= S.t0)
                throw std::runtime_error("segment " + std::to_string(S.k) + ": no generated tokens past t0=" + std::to_string(S.t0));

            const Json& st = j["stages"];
            for (size_t s = 0; s < st.size(); ++s) {
                const Json& js = st.at(s);
                StageDef SD;
                SD.name = js.str_or("name", "stage_" + std::to_string(s));
                SD.label = js.str_or("label", SD.name);
                SD.group = js.str_or("group", SD.label);
                SD.kind = js.str_or("kind", "layer");
                SD.token_source = js.str_or("token_source", "none");
                SD.count = js["count"].i();
                SD.token_offset = js.i_or("token_offset", 0);
                SD.span_y = static_cast<float>(js.num_or("span_y", 150.0));
                SD.z_jitter = static_cast<float>(js.num_or("z_jitter", 44.0));
                SD.x_jitter = static_cast<float>(js.num_or("x_jitter", 5.0));
                SD.gap_before = static_cast<float>(js.num_or("gap_before", 0.0));
                if (js.has("act") && !js["act"].is_null()) SD.act = map_.arr(js["act"]);
                S.stages.push_back(std::move(SD));
            }
            if (j.has("prefill_links") && !j["prefill_links"].is_null()) {
                const Json& pl = j["prefill_links"];
                for (size_t e = 0; e < pl.size(); ++e) {
                    const Json& je = pl.at(e);
                    S.prefill_links.push_back({je["src"].i(), je["dst"].i(), je.str_or("kind", "self"), je.str_or("tint", "violet"),
                                               map_.arr(je["ids"]), map_.arr(je["weights"])});
                }
            }

            const Json& dc = j["decode"];
            S.in_stage = dc["in_stage"].i();
            S.out_stage = dc["out_stage"].i();
            S.cross_stage = dc.i_or("cross_stage", -1);
            S.layer_stages = json_ints(dc["layer_stages"]);

            const Json& tp = j["tape"];
            S.tape.top_id = map_.arr(tp["top_id"]);
            S.tape.top_p = map_.arr(tp["top_p"]);
            S.tape.entropy = map_.arr(tp["entropy"]);
            if (tp.has("p_ref") && !tp["p_ref"].is_null()) S.tape.p_ref = map_.arr(tp["p_ref"]);
            S.tape.self_id = map_.arr(tp["self_id"]);
            S.tape.self_w = map_.arr(tp["self_w"]);
            if (tp.has("cross_id") && !tp["cross_id"].is_null()) S.tape.cross_id = map_.arr(tp["cross_id"]);
            if (tp.has("cross_w") && !tp["cross_w"].is_null()) S.tape.cross_w = map_.arr(tp["cross_w"]);
            S.tape.norm = map_.arr(tp["norm"]);

            size_t m = S.ftok.n;
            if (S.tape.top_id.dim(0) != m || S.tape.self_id.dim(0) != m || S.tape.norm.dim(0) != m)
                throw std::runtime_error("segment " + std::to_string(S.k) + ": tape dimensions do not match generated length");

            S.start = t;
            S.write_ticks = static_cast<double>(S.free_n() - S.t0);
            t += S.read_ticks + S.write_ticks;
            segments.push_back(std::move(S));
        }
        total = t;
    }
};

}  // namespace vitro
