#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "Uso: %s --model <gguf> --prompt \"<texto>\" --gen <N> [--state <buffer.bin>] [--state-name <nombre>]\n", prog);
}

static std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if ((unsigned char)c < 0x20) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

static std::string detok(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::vector<char> bigbuf(-n + 1);
        int n2 = llama_token_to_piece(vocab, t, bigbuf.data(), (int)bigbuf.size(), 0, true);
        if (n2 > 0) {
            return std::string(bigbuf.data(), n2);
        }
        return {};
    }
    return std::string(buf, n);
}

struct TokenLogit {
    int id;
    float logit;
};

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt     = nullptr;
    int gen_tokens          = 0;
    const char * state_path = nullptr;
    const char * state_name = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "Falta valor para %s\n", name);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--model")             model_path = next("--model");
        else if (a == "--prompt")       prompt     = next("--prompt");
        else if (a == "--gen")          gen_tokens = atoi(next("--gen"));
        else if (a == "--state")        state_path = next("--state");
        else if (a == "--state-name")   state_name = next("--state-name");
        else {
            fprintf(stderr, "Argumento desconocido: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !prompt || gen_tokens <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    std::string estado_str = "baseline";
    if (state_name) {
        estado_str = state_name;
    } else if (state_path) {
        estado_str = state_path;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU SIEMPRE

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "Error al cargar modelo GGUF: %s\n", model_path);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 256;
    cp.n_batch   = 256;
    cp.n_seq_max = 1;
    cp.n_threads = 8;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "Error al inicializar llama_context\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    int32_t cell_pos = -1;
    int32_t start_pos = 0;
    size_t set_data_bytes = 0;

    if (state_path) {
        FILE * f = fopen(state_path, "rb");
        if (!f) {
            fprintf(stderr, "No se puede abrir el estado: %s\n", state_path);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (fsize < 16) {
            fprintf(stderr, "Estado demasiado chico (%ld bytes)\n", fsize);
            fclose(f);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        std::vector<uint8_t> state_buf(fsize);
        if (fread(state_buf.data(), 1, fsize, f) != (size_t)fsize) {
            fprintf(stderr, "Error leyendo estado: %s\n", state_path);
            fclose(f);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        fclose(f);

        std::memcpy(&cell_pos, state_buf.data() + 12, sizeof(int32_t));
        set_data_bytes = llama_state_seq_set_data(ctx, state_buf.data(), state_buf.size(), 0);
        if (set_data_bytes == 0) {
            printf("set_data_failed\n");
            fprintf(stderr, "set_data_failed\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 3;
        }
        start_pos = cell_pos + 1;
    }

    // Tokenizar prompt
    int n_prompt = llama_tokenize(vocab, prompt, (int)strlen(prompt), nullptr, 0, true, true);
    if (n_prompt < 0) {
        n_prompt = -n_prompt;
    }
    std::vector<llama_token> prompt_tokens(n_prompt);
    int tok_res = llama_tokenize(vocab, prompt, (int)strlen(prompt), prompt_tokens.data(), (int)prompt_tokens.size(), true, true);
    if (tok_res <= 0) {
        fprintf(stderr, "Error tokenizando prompt\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    prompt_tokens.resize(tok_res);
    n_prompt = tok_res;

    // Decodificar prompt en una sola batch desde start_pos
    llama_batch pb = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        pb.token[i]     = prompt_tokens[i];
        pb.pos[i]       = start_pos + i;
        pb.n_seq_id[i]  = 1;
        pb.seq_id[i][0] = 0;
        pb.logits[i]    = (i == n_prompt - 1) ? 1 : 0;
    }
    pb.n_tokens = n_prompt;

    if (llama_decode(ctx, pb) != 0) {
        llama_batch_free(pb);
        fprintf(stderr, "Error decodificando prompt con llama_decode\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    llama_batch_free(pb);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * step1_logits = llama_get_logits_ith(ctx, -1);
    if (!step1_logits) {
        fprintf(stderr, "Error obteniendo logits paso 1\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    float logits_max = -1e30f;
    double logits_suma = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        float val = step1_logits[i];
        if (val > logits_max) {
            logits_max = val;
        }
        logits_suma += val;
    }

    // Top 5 del primer paso
    std::vector<TokenLogit> all_logits(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        all_logits[i] = {i, step1_logits[i]};
    }
    int top_n = std::min(5, n_vocab);
    std::partial_sort(all_logits.begin(), all_logits.begin() + top_n, all_logits.end(),
                      [](const TokenLogit & a, const TokenLogit & b) {
                          return a.logit > b.logit;
                      });
    std::vector<TokenLogit> top5(all_logits.begin(), all_logits.begin() + top_n);

    printf("top5:");
    for (const auto & item : top5) {
        printf(" %d:%.4f", item.id, item.logit);
    }
    printf("\n");
    printf("logits_suma: %.4f\n", logits_suma);
    printf("logits_max: %.4f\n", logits_max);

    // Generar gen_tokens de forma greedy
    std::vector<int> gen_ids;
    std::string continuacion = "";
    std::vector<float> cur_logits(step1_logits, step1_logits + n_vocab);

    for (int k = 0; k < gen_tokens; k++) {
        int best_token = 0;
        float best_logit = cur_logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (cur_logits[i] > best_logit) {
                best_logit = cur_logits[i];
                best_token = i;
            }
        }
        gen_ids.push_back(best_token);
        continuacion += detok(vocab, best_token);

        if (k + 1 < gen_tokens) {
            llama_batch gb = llama_batch_init(1, 0, 1);
            gb.token[0]     = best_token;
            gb.pos[0]       = start_pos + n_prompt + k;
            gb.n_seq_id[0]  = 1;
            gb.seq_id[0][0] = 0;
            gb.logits[0]    = 1;
            gb.n_tokens     = 1;

            if (llama_decode(ctx, gb) != 0) {
                llama_batch_free(gb);
                fprintf(stderr, "Error decodificando token %d en paso %d\n", best_token, k);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 2;
            }
            llama_batch_free(gb);

            const float * next_logits = llama_get_logits_ith(ctx, -1);
            if (!next_logits) {
                fprintf(stderr, "Error obteniendo logits en paso %d\n", k + 1);
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 2;
            }
            cur_logits.assign(next_logits, next_logits + n_vocab);
        }
    }

    printf("ids:");
    for (int id : gen_ids) {
        printf(" %d", id);
    }
    printf("\n");
    printf("continuacion: %s\n", continuacion.c_str());

    // RESULT_JSON en una sola línea
    std::string json = "RESULT_JSON: {";
    json += "\"estado\": \"" + json_escape(estado_str) + "\", ";
    json += "\"set_data_bytes\": " + std::to_string(set_data_bytes) + ", ";
    json += "\"cell_pos\": " + std::to_string(cell_pos) + ", ";
    json += "\"start_pos\": " + std::to_string(start_pos) + ", ";
    json += "\"prompt_tokens\": " + std::to_string(n_prompt) + ", ";
    json += "\"top5\": [";
    for (size_t i = 0; i < top5.size(); i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "[%d, %.4f]", top5[i].id, top5[i].logit);
        json += buf;
        if (i + 1 < top5.size()) json += ", ";
    }
    json += "], ";
    char max_buf[64], sum_buf[64];
    snprintf(max_buf, sizeof(max_buf), "%.4f", logits_max);
    snprintf(sum_buf, sizeof(sum_buf), "%.4f", logits_suma);
    json += "\"logits_max\": " + std::string(max_buf) + ", ";
    json += "\"logits_suma\": " + std::string(sum_buf) + ", ";
    json += "\"ids\": [";
    for (size_t i = 0; i < gen_ids.size(); i++) {
        json += std::to_string(gen_ids[i]);
        if (i + 1 < gen_ids.size()) json += ", ";
    }
    json += "], ";
    json += "\"continuacion\": \"" + json_escape(continuacion) + "\"}";

    printf("%s\n", json.c_str());
    fflush(stdout);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
