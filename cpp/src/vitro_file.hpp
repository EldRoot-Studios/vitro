// Reads a Vitro trace dump (.vtdump, magic "VITRO01\n", kind "dump").
// Supports arbitrary model architectures via dynamic stages, links, attention families, spans, and user tags.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <vitro/container.hpp>

namespace vitro {

struct SpanInfo {
    int id = 0;
    int start = 0, end = 0;
    int parent = -1;
    std::string label, kind, tint, meta;
    std::vector<int> children;
};

struct StageDef {
    std::string name, label, group, kind;      // kind: "input", "layer", "latent", "output"
    std::string token_source;                   // "none", "source", "target"
    int count = 0;
    int token_offset = 0;
    float span_y = 150.0f, z_jitter = 44.0f, x_jitter = 5.0f, gap_before = 0.0f;
    Arr act;                                    // optional (count,) float
};

struct LinkDef {
    int src = 0, dst = 0, layer_idx = -1, row_offset = 0;
    std::string name, family, kind, tint;       // kind: "resid", "self", "cross", "out"
    bool causal = false;
    Arr weights;                                // optional (H, R, C) or (R, C)
};

struct AttnFamily {
    std::string name, row_label, col_label;
    std::vector<int> link_indices;              // per layer (index into UnitTrace::links)
    int layers = 0, heads = 0, rows = 0, cols = 0, row_offset = 0;
    bool causal = false;
};

struct NormBlock {
    std::string name;
    std::vector<std::string> layers, components;
    Arr data;                                   // (L, C, R)
    Arr in_norm;                                // optional (R,)
    int row_offset = 0;
};

struct LatentMap {
    std::string name, row_label, col_label;
    Arr data;                                   // (S, D)
    Arr norms;                                  // optional (S,)
};

struct UnitTrace {
    int u = 0;
    std::string label, kind;
    int t0 = 1;
    double bits = 0.0;
    bool forced_exact = false, free_exact = false, free_ended = false;
    int diverge = -1, anomalies = 0;

    Arr tok;                                    // (T_out,) target sequence token ids
    Arr src_tok;                                // (E_src,) optional encoder/source token ids
    Arr token_tags;                             // optional (T_out,) tag index
    Arr p, rank, top_id, top_p, entropy, n_allowed, right;
    Arr val_mean, val_std;                      // optional (T_out,) scalar error metrics
    Arr ftok, fp;                               // optional free-generation rollout

    std::vector<StageDef> stages;
    std::vector<LinkDef> links;
    std::vector<AttnFamily> attn_families;
    NormBlock norms;
    std::vector<LatentMap> latents;

    int n_tokens() const { return static_cast<int>(tok.n); }
    int n_source() const { return src_tok.empty() ? n_tokens() : static_cast<int>(src_tok.n); }
    int free_n() const { return static_cast<int>(ftok.n); }
};

class TraceDump {
public:
    TraceDump() = default;
    explicit TraceDump(const std::string& p) { open(p); }
    TraceDump(TraceDump&&) = default;
    TraceDump& operator=(TraceDump&&) = default;
    TraceDump(const TraceDump&) = delete;
    TraceDump& operator=(const TraceDump&) = delete;

    bool loaded() const { return !units.empty(); }
    size_t data_size() const { return map_.data_size; }

    std::string path;
    Json head;
    std::string uid, title, model_name, arch, precision, device;
    long long step = -1;
    int top_k = 5;
    double delta_red_at = 12.0;

    std::vector<std::string> labels, tag_names;
    Arr token_values;
    std::vector<SpanInfo> spans;
    std::vector<UnitTrace> units;

    const std::string& label(int id) const {
        if (id < 0 || id >= static_cast<int>(labels.size()))
            throw std::runtime_error("token id " + std::to_string(id) + " outside vocabulary");
        return labels[static_cast<size_t>(id)];
    }
    const std::string& tag_name(int tag_id) const {
        static const std::string empty;
        if (tag_id < 0 || tag_id >= static_cast<int>(tag_names.size())) return empty;
        return tag_names[static_cast<size_t>(tag_id)];
    }
    double value_of(int id) const {
        if (id < 0 || static_cast<size_t>(id) >= token_values.n) return -1.0;
        return static_cast<double>(token_values.f(static_cast<size_t>(id)));
    }
    bool has_value(int id) const {
        double v = value_of(id);
        return !std::isnan(v) && v >= 0.0;
    }

private:
    MappedContainer map_;

    static void build_attn_families(UnitTrace& u) {
        for (size_t i = 0; i < u.links.size(); ++i) {
            const LinkDef& lk = u.links[i];
            if (lk.weights.empty() || lk.family.empty()) continue;
            AttnFamily* fam = nullptr;
            for (AttnFamily& f : u.attn_families)
                if (f.name == lk.family) { fam = &f; break; }
            if (!fam) {
                u.attn_families.push_back(AttnFamily{});
                fam = &u.attn_families.back();
                fam->name = lk.family;
                fam->causal = lk.causal;
                fam->row_offset = lk.row_offset;
                const StageDef& ds = u.stages[static_cast<size_t>(lk.dst)];
                const StageDef& ss = u.stages[static_cast<size_t>(lk.src)];
                fam->row_label = ds.label.empty() ? "Query position" : ds.label;
                fam->col_label = ss.label.empty() ? "Key position" : ss.label;
                if (lk.weights.shape.size() == 3) {
                    fam->heads = static_cast<int>(lk.weights.dim(0));
                    fam->rows = static_cast<int>(lk.weights.dim(1));
                    fam->cols = static_cast<int>(lk.weights.dim(2));
                } else if (lk.weights.shape.size() == 2) {
                    fam->heads = 1;
                    fam->rows = static_cast<int>(lk.weights.dim(0));
                    fam->cols = static_cast<int>(lk.weights.dim(1));
                } else {
                    throw std::runtime_error("attention weights must be 2D or 3D");
                }
            } else {
                const int h = lk.weights.shape.size() == 3 ? static_cast<int>(lk.weights.dim(0)) : 1;
                const int r = static_cast<int>(lk.weights.dim(lk.weights.shape.size() == 3 ? 1 : 0));
                const int c = static_cast<int>(lk.weights.dim(lk.weights.shape.size() == 3 ? 2 : 1));
                if (h != fam->heads || r != fam->rows || c != fam->cols)
                    throw std::runtime_error("attention family '" + fam->name + "' has mismatched layer shapes");
            }
            fam->link_indices.push_back(static_cast<int>(i));
            fam->layers = static_cast<int>(fam->link_indices.size());
        }
    }

    void open(const std::string& p) {
        path = p;
        map_.open(p, "VITRO01\n", "not a Vitro trace file (expected VITRO01 magic)");
        head = map_.head;
        const std::string kind = head.str_or("kind", "dump");
        if (kind != "dump")
            throw std::runtime_error(p + ": this is a '" + kind + "' recording, open it with vitro_demo");

        const Json& tr = head["trace"];
        uid = tr.str_or("id", "trace");
        title = tr.str_or("title", uid);
        model_name = tr.str_or("model_name", "PyTorch Model");
        arch = tr.str_or("arch", "custom");
        precision = tr.str_or("precision", "fp32");
        device = tr.str_or("device", "cpu");
        step = tr.has("step") && !tr["step"].is_null() ? static_cast<long long>(tr["step"].num()) : -1;
        top_k = tr.i_or("top_k", 5);
        delta_red_at = tr.num_or("delta_red_at", 12.0);

        const Json& vb = head["vocab"];
        labels = json_strings(vb["labels"]);
        if (vb.has("tag_names") && !vb["tag_names"].is_null()) tag_names = json_strings(vb["tag_names"]);
        if (vb.has("values") && !vb["values"].is_null()) token_values = map_.arr(vb["values"]);

        if (head.has("spans") && !head["spans"].is_null()) {
            const Json& sp = head["spans"];
            spans.reserve(sp.size());
            for (size_t i = 0; i < sp.size(); ++i) {
                const Json& js = sp.at(i);
                SpanInfo S;
                S.id = js.i_or("id", static_cast<int>(i));
                S.start = js["start"].i();
                S.end = js["end"].i();
                S.parent = js.i_or("parent", -1);
                S.label = js["label"].str();
                S.kind = js.str_or("kind", "region");
                S.tint = js.str_or("tint", "");
                S.meta = js.str_or("meta", "");
                spans.push_back(std::move(S));
            }
            for (size_t i = 0; i < spans.size(); ++i) {
                int par = spans[i].parent;
                if (par >= 0 && par < static_cast<int>(spans.size()) && par != static_cast<int>(i))
                    spans[static_cast<size_t>(par)].children.push_back(static_cast<int>(i));
            }
        }

        const Json& us = head["units"];
        if (us.size() == 0) throw std::runtime_error(p + ": trace contains zero units");
        for (size_t ui = 0; ui < us.size(); ++ui) {
            const Json& j = us.at(ui);
            UnitTrace U;
            U.u = j.i_or("u", static_cast<int>(ui));
            U.label = j.str_or("label", "Unit " + std::to_string(U.u));
            U.kind = j.str_or("kind", "sequence");
            U.t0 = j.i_or("t0", 1);
            U.bits = j.num_or("bits", 0.0);
            U.forced_exact = j.bool_or("forced_exact", false);
            U.free_exact = j.bool_or("free_exact", false);
            U.free_ended = j.bool_or("free_ended", false);
            U.diverge = j.i_or("diverge", -1);
            U.anomalies = j.i_or("anomalies", 0);

            const Json& seq = j["sequence"];
            U.tok = map_.arr(seq["tokens"]);
            if (seq.has("source_tokens") && !seq["source_tokens"].is_null()) U.src_tok = map_.arr(seq["source_tokens"]);
            if (seq.has("token_tags") && !seq["token_tags"].is_null()) U.token_tags = map_.arr(seq["token_tags"]);
            U.p = map_.arr(seq["p"]);
            U.rank = map_.arr(seq["rank"]);
            U.top_id = map_.arr(seq["top_id"]);
            U.top_p = map_.arr(seq["top_p"]);
            U.entropy = map_.arr(seq["entropy"]);
            if (seq.has("n_allowed") && !seq["n_allowed"].is_null()) U.n_allowed = map_.arr(seq["n_allowed"]);
            U.right = map_.arr(seq["right"]);
            if (seq.has("val_mean") && !seq["val_mean"].is_null()) U.val_mean = map_.arr(seq["val_mean"]);
            if (seq.has("val_std") && !seq["val_std"].is_null()) U.val_std = map_.arr(seq["val_std"]);
            if (seq.has("free_tok") && !seq["free_tok"].is_null()) U.ftok = map_.arr(seq["free_tok"]);
            if (seq.has("free_p") && !seq["free_p"].is_null()) U.fp = map_.arr(seq["free_p"]);

            if (U.p.n != U.tok.n || U.rank.n != U.tok.n || U.entropy.n != U.tok.n)
                throw std::runtime_error("unit " + std::to_string(U.u) + ": per-token metric shapes do not match tokens length");

            const Json& st = j["stages"];
            for (size_t s = 0; s < st.size(); ++s) {
                const Json& js = st.at(s);
                StageDef S;
                S.name = js.str_or("name", "stage_" + std::to_string(s));
                S.label = js.str_or("label", S.name);
                S.group = js.str_or("group", S.label);
                S.kind = js.str_or("kind", "layer");
                S.token_source = js.str_or("token_source", "none");
                S.count = js["count"].i();
                S.token_offset = js.i_or("token_offset", 0);
                S.span_y = static_cast<float>(js.num_or("span_y", 150.0));
                S.z_jitter = static_cast<float>(js.num_or("z_jitter", 44.0));
                S.x_jitter = static_cast<float>(js.num_or("x_jitter", 5.0));
                S.gap_before = static_cast<float>(js.num_or("gap_before", 0.0));
                if (js.has("act") && !js["act"].is_null()) S.act = map_.arr(js["act"]);
                U.stages.push_back(std::move(S));
            }

            const Json& lk = j["links"];
            for (size_t e = 0; e < lk.size(); ++e) {
                const Json& je = lk.at(e);
                LinkDef L;
                L.src = je["src"].i();
                L.dst = je["dst"].i();
                if (L.src < 0 || L.src >= static_cast<int>(U.stages.size()) || L.dst < 0 || L.dst >= static_cast<int>(U.stages.size()))
                    throw std::runtime_error("unit " + std::to_string(U.u) + ": link stage index out of range");
                L.name = je.str_or("name", "");
                L.family = je.str_or("family", "");
                L.kind = je.str_or("kind", "self");
                L.tint = je.str_or("tint", "violet");
                L.layer_idx = je.i_or("layer_idx", -1);
                L.row_offset = je.i_or("row_offset", 0);
                L.causal = je.bool_or("causal", false);
                if (je.has("weights") && !je["weights"].is_null()) L.weights = map_.arr(je["weights"]);
                U.links.push_back(std::move(L));
            }
            build_attn_families(U);

            if (j.has("norms") && !j["norms"].is_null()) {
                const Json& nm = j["norms"];
                U.norms.name = nm.str_or("name", "Layer Activations");
                U.norms.layers = json_strings(nm["layers"]);
                U.norms.components = json_strings(nm["components"]);
                U.norms.data = map_.arr(nm["data"]);
                if (nm.has("in_norm") && !nm["in_norm"].is_null()) U.norms.in_norm = map_.arr(nm["in_norm"]);
                U.norms.row_offset = nm.i_or("row_offset", 0);
            }
            if (j.has("latents") && !j["latents"].is_null()) {
                const Json& lt = j["latents"];
                for (size_t m = 0; m < lt.size(); ++m) {
                    const Json& jm = lt.at(m);
                    LatentMap M;
                    M.name = jm["name"].str();
                    M.row_label = jm.str_or("row_label", "Slot");
                    M.col_label = jm.str_or("col_label", "Dim");
                    M.data = map_.arr(jm["data"]);
                    if (jm.has("norms") && !jm["norms"].is_null()) M.norms = map_.arr(jm["norms"]);
                    U.latents.push_back(std::move(M));
                }
            }
            units.push_back(std::move(U));
        }
    }
};

}  // namespace vitro
