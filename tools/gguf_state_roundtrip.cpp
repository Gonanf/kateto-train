// B1 experiment: can a GGUF model accept an external state?
// Three tests, in order: (1) byte identity of the state buffer, (2) logits equality,
// (3) greedy continuation equality. See docs/gguf-estado-roundtrip.md.
#include <llama.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char *MODEL_DEFAULT = "out/kateto-rwkv29-Q4_K_M.gguf";
static const char *PROMPT_DEFAULT = "Hola, soy Kateto.";

static void die_api(const char *what) {
    fprintf(stderr, "\nAPI_FAIL: %s returned failure\n", what);
    exit(2);  // llama_log already printed the raw reason to stderr
}

static void die(const char *fmt, const char *arg) {
    fprintf(stderr, "\nFATAL: ");
    fprintf(stderr, fmt, arg);
    fprintf(stderr, "\n");
    exit(2);
}

// Decode one token into an explicit sequence. llama_batch_get_one hardcodes seq_id 0,
// so seq 1 needs a hand-built batch.
static void decode_one(llama_context * ctx, llama_token t, int pos, llama_seq_id seq) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0]    = t;
    b.pos[0]      = pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = seq;
    b.logits[0]   = 1;
    b.n_tokens    = 1;
    if (llama_decode(ctx, b) != 0) {
        llama_batch_free(b);
        die_api("llama_decode");
    }
    llama_batch_free(b);
}

static const float * logits_after(llama_context * ctx, int idx) {
    const float * l = llama_get_logits_ith(ctx, idx);
    if (!l) {
        die("no logits at batch index %d", "out of range");
    }
    return l;
}

static int argmax_of(const float * l, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (l[i] > l[best]) {
            best = i;
        }
    }
    return best;
}

static std::string detok(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return {};
    }
    return std::string(buf, n);
}

int main(int argc, char ** argv) {
    const char * model_path = MODEL_DEFAULT;
    const char * prompt     = PROMPT_DEFAULT;
    const char * dump_path  = nullptr;
    const char * inject     = nullptr;
    bool vulkan             = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next     = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                die("missing value for %s", name);
            }
            return argv[++i];
        };
        if (a == "--model")        model_path = next("--model");
        else if (a == "--prompt")  prompt     = next("--prompt");
        else if (a == "--dump-state")  dump_path = next("--dump-state");
        else if (a == "--inject-state") inject  = next("--inject-state");
        else if (a == "--backend") {
            std::string b = next("--backend");
            vulkan = (b == "vulkan");
            if (!vulkan && b != "cpu") {
                die("unknown backend %s (use cpu or vulkan)", b.c_str());
            }
        } else {
            die("unknown arg %s (see --model --prompt --backend --dump-state --inject-state)", a.c_str());
        }
    }

    printf("backend: %s\n", vulkan ? "vulkan" : "cpu");
    printf("model: %s\n", model_path);

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers      = vulkan ? 99 : 0;

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        die_api("llama_model_load_from_file");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp   = llama_context_default_params();
    cp.n_ctx                 = 128;
    cp.n_batch               = 128;
    cp.n_seq_max             = 2;
    cp.n_threads             = 8;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        die_api("llama_init_from_model");
    }

    // ---- step 2: tokenize + decode the prompt on seq 0 ----
    std::vector<llama_token> toks(128);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), 128, true, true);
    if (n <= 0) {
        die("tokenize produced %d tokens for prompt", "prompt");
    }
    printf("prompt: %s\nprompt_tokens: %d\n", prompt, n);

    llama_batch pb = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        pb.token[i]    = toks[i];
        pb.pos[i]      = i;
        pb.n_seq_id[i] = 1;
        pb.seq_id[i][0] = 0;
        pb.logits[i]   = (i == n - 1) ? 1 : 0;
    }
    pb.n_tokens = n;
    if (llama_decode(ctx, pb) != 0) {
        llama_batch_free(pb);
        die_api("llama_decode(prompt)");
    }
    llama_batch_free(pb);

    const int n_vocab = llama_vocab_n_tokens(vocab);

    // ---- step 3: grab the state of seq 0 ----
    size_t sz = llama_state_seq_get_size(ctx, 0);
    if (sz == 0) {
        die_api("llama_state_seq_get_size");
    }
    printf("size_bytes: %zu\n", sz);

    std::vector<uint8_t> A(sz);
    if (inject) {
        FILE * f = fopen(inject, "rb");
        if (!f) {
            die("cannot open %s for reading", inject);
        }
        size_t got = fread(A.data(), 1, sz, f);
        fclose(f);
        if (got != sz) {
            die("state file holds %zu bytes, expected %zu", "file");
        }
        printf("injected_state: %s\n", inject);
    } else {
        llama_state_seq_get_data(ctx, A.data(), sz, 0);
        if (dump_path) {
            FILE * f = fopen(dump_path, "wb");
            if (!f) {
                die("cannot open %s for writing", dump_path);
            }
            fwrite(A.data(), 1, sz, f);
            fclose(f);
            printf("dumped_state: %s\n", dump_path);
        }
    }

    bool t1_ok = false, t2_ok = false, t3_ok = false;
    float max_abs_diff = 0.0f;
    int argmax_0 = -1, argmax_1 = -1;

    // ---- test 1: byte identity ----
    size_t written = llama_state_seq_set_data(ctx, A.data(), sz, 1);
    printf("state_seq_set_data_bytes: %zu\n", written);
    if (written == 0) {
        printf("test1_identidad: FAIL (set_data on seq 1 returned 0)\n");
    } else {
        std::vector<uint8_t> B(sz);
        llama_state_seq_get_data(ctx, B.data(), sz, 1);

        // The first 8 bytes are the stream header: uint32 magic + int32 seq_id.
        // set_data reads them but restores into the seq_id passed as argument, so
        // the seq_id field legitimately differs between a seq-0 and a seq-1 dump.
        const size_t hdr = sizeof(uint32_t) + sizeof(int32_t);
        printf("header_seq_id_0: %d\nheader_seq_id_1: %d\n",
               (int)(*(const int32_t *)(A.data() + sizeof(uint32_t))),
               (int)(*(const int32_t *)(B.data() + sizeof(uint32_t))));

        long long first = -1;
        for (size_t i = hdr; i < sz; i++) {
            if (A[i] != B[i]) {
                first = (long long)i;
                break;
            }
        }
        if (first < 0) {
            printf("test1_identidad: OK (%zu payload bytes identical after %zu header bytes)\n",
                   sz - hdr, hdr);
            t1_ok = true;
        } else {
            printf("test1_identidad: FAIL (first payload diff at byte %lld: %02x vs %02x)\n",
                   first, A[first], B[first]);
        }
    }

    // ---- test 2: same next token on seq 0 and seq 1 ----
    llama_token next_tok = toks[n - 1];  // repeat the last prompt token
    // llama_get_logits_ith points into the context's internal buffer, so both
    // vectors must be copied before the next decode overwrites them.
    decode_one(ctx, next_tok, n, 0);
    std::vector<float> l0(logits_after(ctx, 0), logits_after(ctx, 0) + n_vocab);
    decode_one(ctx, next_tok, n, 1);
    std::vector<float> l1(logits_after(ctx, 0), logits_after(ctx, 0) + n_vocab);

    for (int i = 0; i < n_vocab; i++) {
        float d = fabsf(l0[i] - l1[i]);
        if (d > max_abs_diff) {
            max_abs_diff = d;
        }
    }
    argmax_0 = argmax_of(l0.data(), n_vocab);
    argmax_1 = argmax_of(l1.data(), n_vocab);
    printf("max_abs_diff: %g\n", max_abs_diff);
    printf("argmax_0: %d\nargmax_1: %d\n", argmax_0, argmax_1);
    printf("argmax_igual: %s\n", argmax_0 == argmax_1 ? "si" : "no");
    t2_ok = (max_abs_diff <= 1e-3f) && (argmax_0 == argmax_1);
    printf("test2_logits: %s\n", t2_ok ? "OK" : "FAIL");

    // ---- test 3: 8 greedy tokens per sequence ----
    std::string c0, c1;
    {
        std::vector<float> cur = l0;
        for (int k = 0; k < 8; k++) {
            llama_token t = argmax_of(cur.data(), n_vocab);
            c0 += detok(vocab, t);
            decode_one(ctx, t, n + 1 + k, 0);
            cur.assign(logits_after(ctx, 0), logits_after(ctx, 0) + n_vocab);
        }
    }
    {
        std::vector<float> cur = l1;
        for (int k = 0; k < 8; k++) {
            llama_token t = argmax_of(cur.data(), n_vocab);
            c1 += detok(vocab, t);
            decode_one(ctx, t, n + 1 + k, 1);
            cur.assign(logits_after(ctx, 0), logits_after(ctx, 0) + n_vocab);
        }
    }
    printf("continuacion_seq0: %s\n", c0.c_str());
    printf("continuacion_seq1: %s\n", c1.c_str());
    t3_ok = (c0 == c1);
    printf("continuacion_igual: %s\n", t3_ok ? "si" : "no");
    printf("test3_continuacion: %s\n", t3_ok ? "OK" : "FAIL");

    bool pass = t1_ok && t2_ok && t3_ok;
    printf("RESULTADO: %s\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("fallos: test1=%s test2=%s test3=%s\n",
               t1_ok ? "OK" : "FAIL", t2_ok ? "OK" : "FAIL", t3_ok ? "OK" : "FAIL");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 1;
}