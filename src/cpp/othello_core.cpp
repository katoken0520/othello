#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <random>
#include <fstream>
#include <atomic>

namespace py = pybind11;

#ifdef _MSC_VER
    #include <intrin.h>
    #define POPCOUNT(x) __popcnt64(x)
    inline int GET_LSB_IDX(uint64_t x) {
        unsigned long idx;
        _BitScanForward64(&idx, x);
        return (int)idx;
    }
#else
    #define POPCOUNT(x) __builtin_popcountll(x)
    #define GET_LSB_IDX(x) __builtin_ctzll(x)
#endif

const uint64_t MASK_HOR     = 0x7e7e7e7e7e7e7e7eULL;
const uint64_t MASK_VER     = 0xffffffffffffffffULL;
const uint64_t MASK_HOR_VER = 0x7e7e7e7e7e7e7e7eULL;
const float STATIC_MOVE_SCORE[64] = {
    100.0f, -50.0f,  20.0f,   5.0f,   5.0f,  20.0f, -50.0f, 100.0f,
    -50.0f, -100.0f, -5.0f,  -5.0f,  -5.0f,  -5.0f, -100.0f,-50.0f,
     20.0f,  -5.0f,  15.0f,   3.0f,   3.0f,  15.0f,  -5.0f,  20.0f,
      5.0f,  -5.0f,   3.0f,   3.0f,   3.0f,   3.0f,  -5.0f,   5.0f,
      5.0f,  -5.0f,   3.0f,   3.0f,   3.0f,   3.0f,  -5.0f,   5.0f,
     20.0f,  -5.0f,  15.0f,   3.0f,   3.0f,  15.0f,  -5.0f,  20.0f,
    -50.0f, -100.0f, -5.0f,  -5.0f,  -5.0f,  -5.0f, -100.0f,-50.0f,
    100.0f, -50.0f,  20.0f,   5.0f,   5.0f,  20.0f, -50.0f, 100.0f
};

uint64_t _legal(uint64_t player, uint64_t op_masked, uint64_t empty, int shift) {
    uint64_t candidates_l = (player << shift) & op_masked;
    uint64_t candidates_r = (player >> shift) & op_masked;
    for (int i = 0; i < 5; ++i) {
        candidates_l |= (candidates_l << shift) & op_masked;
        candidates_r |= (candidates_r >> shift) & op_masked;
    }
    return ((candidates_l << shift) | (candidates_r >> shift)) & empty;
}

uint64_t get_valid_moves(uint64_t player, uint64_t opponent) {
    uint64_t empty = ~(player | opponent);
    uint64_t op_h  = opponent & MASK_HOR;
    uint64_t op_v  = opponent & MASK_VER;
    uint64_t op_hv = opponent & MASK_HOR_VER;

    uint64_t legal = _legal(player, op_h, empty, 1);
    legal |= _legal(player, op_v, empty, 8);
    legal |= _legal(player, op_hv, empty, 7);
    legal |= _legal(player, op_hv, empty, 9);
    return legal;
}

uint64_t _flip(uint64_t player, uint64_t op_masked, uint64_t put, int shift) {
    uint64_t tmp_l = (put << shift) & op_masked;
    uint64_t tmp_r = (put >> shift) & op_masked;
    uint64_t flip_bit_l = tmp_l;
    uint64_t flip_bit_r = tmp_r;
    uint64_t flip_bit = 0ULL;

    if (flip_bit_l != 0ULL) {
        for (int i = 0; i < 6; ++i) {
            tmp_l = tmp_l << shift;
            if ((tmp_l & op_masked) == 0ULL) break;
            flip_bit_l |= tmp_l;
        }
        if ((player & tmp_l) != 0ULL) flip_bit |= flip_bit_l;
    }
    if (flip_bit_r != 0ULL) {
        for (int i = 0; i < 6; ++i) {
            tmp_r = tmp_r >> shift;
            if ((tmp_r & op_masked) == 0ULL) break;
            flip_bit_r |= tmp_r;
        }
        if ((player & tmp_r) != 0ULL) flip_bit |= flip_bit_r;
    }
    return flip_bit;
}

uint64_t get_flipped_bb(uint64_t put_pos_bit, uint64_t player, uint64_t opponent) {
    uint64_t op_h  = opponent & MASK_HOR;
    uint64_t op_v  = opponent & MASK_VER;
    uint64_t op_hv = opponent & MASK_HOR_VER;
    uint64_t flipped = _flip(player, op_h, put_pos_bit, 1);
    flipped |= _flip(player, op_v, put_pos_bit, 8);
    flipped |= _flip(player, op_hv, put_pos_bit, 7);
    flipped |= _flip(player, op_hv, put_pos_bit, 9);
    return flipped;
}

enum HashFlag : uint8_t { HASH_EXACT, HASH_ALPHA, HASH_BETA };
const int MAX_TT_MOVES = 3;

struct TTEntry {
    uint64_t hash;
    float score;
    uint32_t depth : 6;
    uint32_t flag  : 2;
    uint32_t num_moves : 2;
    uint32_t packed_moves : 18;
    uint32_t padding : 4;
};

struct MoveInfo {
    uint64_t move_bit;
    float ordering_score;
    bool operator<(const MoveInfo& other) const {
        return ordering_score > other.ordering_score; 
    }
};

struct RootMove {
    uint64_t put_bit;
    int idx;
    float score;
    bool operator<(const RootMove& other) const { 
        return score > other.score; 
    }
};

class SearchEngine {
private:
    std::vector<float> weights;
    std::vector<std::vector<std::vector<int>>> patterns;
    std::vector<int> pattern_sizes;
    std::vector<int> mlp_layers;
    std::vector<int> active_features;
    int additional_features_num;
    uint64_t zobrist_table[2][64];
    const int TT_SIZE = 1 << 22; 
    const int TT_MASK = (1 << 22) - 1;
    std::vector<TTEntry> tt;
    
    // ★修正: 1局面に対して複数手(手と評価値のペア)を保存できるように変更
    std::unordered_map<uint64_t, std::vector<std::pair<int, float>>> opening_book;
    
    int history_table[64]; 
    std::vector<std::pair<int, float>> last_root_scores;
    std::atomic<bool> abort_search{false};
    const float NULL_WINDOW_EPSILON = 0.1f;

    // ★追加: ランダム選択用の乱数生成器
    std::mt19937_64 rng_engine;

    uint64_t calc_hash(uint64_t player, uint64_t opponent) {
        uint64_t h = 0ULL;
        uint64_t p = player, o = opponent;
        while (p > 0) { h ^= zobrist_table[0][GET_LSB_IDX(p & (~p + 1))]; p &= p - 1; }
        while (o > 0) { h ^= zobrist_table[1][GET_LSB_IDX(o & (~o + 1))]; o &= o - 1; }
        return h;
    }

public:
    SearchEngine() : rng_engine(std::random_device{}()) { // ★乱数器の初期化
        TTEntry empty_entry = {0, 0.0f, 0, HASH_EXACT, {0}, 0};
        tt.resize(TT_SIZE, empty_entry);
        std::mt19937_64 init_rng(42);
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < 64; ++i) zobrist_table[c][i] = init_rng();
        }
    }

    void stop() {
        abort_search = true;
    }

    void clear_tt() {
        TTEntry empty_entry = {0, 0.0f, 0, HASH_EXACT, {0}, 0};
        std::fill(tt.begin(), tt.end(), empty_entry);
    }

    void set_structure(const std::vector<std::vector<std::vector<int>>>& py_patterns, 
                       const std::vector<int>& py_mlp_layers,
                       const std::vector<int>& py_active_features) {
        patterns = py_patterns;
        mlp_layers = py_mlp_layers;
        active_features = py_active_features;
        additional_features_num = (int)active_features.size();
        
        pattern_sizes.clear();
        for (const auto& feature : patterns) {
            int n_bits = (int)feature[0].size();
            int size = 1;
            for (int i = 0; i < n_bits; ++i) size *= 3;
            pattern_sizes.push_back(size);
        }
    }

    void load_weights(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (file) {
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            weights.resize(size / sizeof(float));
            file.read(reinterpret_cast<char*>(weights.data()), size);
        }
    }

    // ★修正: 評価値(score)も一緒に保存できるように引数を追加
    void add_book_move(uint64_t hash, int move_idx, float score) {
        opening_book[hash].push_back({move_idx, score});
    }

    // ★追加: 登録済みの定石をクリアする関数
    void clear_book() {
        opening_book.clear();
    }

    uint64_t get_board_hash(uint64_t player, uint64_t opponent) {
        return calc_hash(player, opponent);
    }

    float evaluate(uint64_t player, uint64_t opponent) {
        if (weights.empty() || patterns.empty()) { 
            return (float)((int)POPCOUNT(player) - (int)POPCOUNT(opponent));
        }

        int input_dim = (int)patterns.size() + additional_features_num;
        std::vector<float> current_activation(input_dim, 0.0f);
        int weight_offset = 0;

        uint64_t empty_bb = ~(player | opponent);
        for (size_t i = 0; i < patterns.size(); ++i) {
            float sum_val = 0.0f;
            for (const auto& bits : patterns[i]) {
                int idx = 0;
                int power = 1;
                
                for (int pos : bits) {
                    int state = (((player >> pos) & 1) << 1) | ((empty_bb >> pos) & 1);
                    idx += state * power;
                    power *= 3;
                }
                sum_val += weights[weight_offset + idx];
            }
            current_activation[i] = sum_val;
            weight_offset += pattern_sizes[i];
        }

        int feat_idx = (int)patterns.size();
        for (int feat_id : active_features) {
            switch(feat_id) {
                case 0: current_activation[feat_idx++] = (float)POPCOUNT(get_valid_moves(player, opponent)); break;
                case 1: current_activation[feat_idx++] = (float)POPCOUNT(get_valid_moves(opponent, player)); break;
                case 2: current_activation[feat_idx++] = (float)POPCOUNT(player); break;
                case 3: current_activation[feat_idx++] = (float)POPCOUNT(opponent); break;
                default: current_activation[feat_idx++] = 0.0f; break; 
            }
        }

        for (size_t l = 0; l < mlp_layers.size() - 1; ++l) {
            int in_dim = mlp_layers[l];
            int out_dim = mlp_layers[l + 1];
            std::vector<float> next_activation(out_dim, 0.0f);

            for (int i = 0; i < out_dim; ++i) {
                float val = weights[weight_offset + out_dim * in_dim + i]; 
                for (int j = 0; j < in_dim; ++j) {
                    val += current_activation[j] * weights[weight_offset + i * in_dim + j];
                }
                if (l < mlp_layers.size() - 2) {
                    next_activation[i] = std::max(0.0f, val);
                } else {
                    next_activation[i] = val;
                }
            }
            weight_offset += out_dim * in_dim + out_dim;
            current_activation = next_activation;
        }

        return current_activation[0];
    }

    float nega_scout(uint64_t player, uint64_t opponent, int depth, bool passed, float alpha, float beta) {
        if (abort_search) return 0.0f;

        uint64_t legal = get_valid_moves(player, opponent);
        if (legal == 0ULL) {
            if (passed) { 
                int diff = (int)POPCOUNT(player) - (int)POPCOUNT(opponent);
                return diff > 0 ? 10000.0f + diff : (diff < 0 ? -10000.0f + diff : 0.0f);
            }
            return -nega_scout(opponent, player, depth, true, -beta, -alpha);
        }

        if (depth == 0) return evaluate(player, opponent);

        uint64_t hash = calc_hash(player, opponent);
        TTEntry& tte = tt[hash & TT_MASK];
        
        uint8_t tt_moves[MAX_TT_MOVES];
        uint8_t tt_num_moves = 0;
        if (tte.hash == hash) {
            tt_num_moves = tte.num_moves;
            for (int i = 0; i < tt_num_moves; ++i) {
                tt_moves[i] = (tte.packed_moves >> (i * 6)) & 0x3F;
            }
            if (tte.depth >= depth) {
                if (tte.flag == HASH_EXACT) return tte.score;
                if (tte.flag == HASH_ALPHA && tte.score <= alpha) return tte.score;
                if (tte.flag == HASH_BETA && tte.score >= beta) return tte.score;
            }
        }

        std::vector<MoveInfo> moves;
        moves.reserve(POPCOUNT(legal));
        while (legal > 0ULL) {
            uint64_t put_bit = legal & (~legal + 1); 
            int idx = GET_LSB_IDX(put_bit);
            
            float order_score = STATIC_MOVE_SCORE[idx] + (float)history_table[idx]; 
            
            for (int i = 0; i < tt_num_moves; ++i) {
                if (tt_moves[i] == idx) {
                    order_score = 100000.0f + (float)(tt_num_moves - i); 
                    break;
                }
            }
            moves.push_back({put_bit, order_score});
            legal ^= put_bit;
        }
        std::sort(moves.begin(), moves.end());

        float original_alpha = alpha;
        float max_score = -1e9f;
        int current_best_move = -1;

        for (int i = 0; i < moves.size(); ++i) {
            if (abort_search) return 0.0f;

            uint64_t put_pos_bit = moves[i].move_bit;
            int current_idx = GET_LSB_IDX(put_pos_bit);
            uint64_t flipped = get_flipped_bb(put_pos_bit, player, opponent);
            uint64_t new_player = player ^ (put_pos_bit | flipped);
            uint64_t new_opponent = opponent ^ flipped;

            float score;
            if (i == 0) {
                score = -nega_scout(new_opponent, new_player, depth - 1, false, -beta, -alpha);
            } else {
                score = -nega_scout(new_opponent, new_player, depth - 1, false, -alpha - NULL_WINDOW_EPSILON, -alpha);
                if (score > alpha && score < beta) {
                    score = -nega_scout(new_opponent, new_player, depth - 1, false, -beta, -alpha);
                }
            }

            if (score > max_score) {
                max_score = score;
                current_best_move = current_idx;
            }
            if (max_score > alpha) {
                alpha = max_score;
            }
            if ((alpha >= beta) && (!abort_search)){
                history_table[current_idx] += depth * depth;
                break;
            }
        }

        if (abort_search) return 0.0f;

        if (current_best_move != -1) {
            history_table[current_best_move] += depth * depth;
        }

        if ((depth >= tte.depth || tte.hash != hash)){
            tte.hash = hash;
            tte.score = max_score;
            tte.depth = (uint8_t)depth; 
            
            if (max_score <= original_alpha) tte.flag = HASH_ALPHA;
            else if (max_score >= beta) tte.flag = HASH_BETA;
            else tte.flag = HASH_EXACT;

            tte.num_moves = 0;
            tte.packed_moves = 0; 
            if (current_best_move != -1) {
                tte.packed_moves |= ((uint32_t)current_best_move & 0x3F);
                tte.num_moves++;
            }
            for (int i = 0; i < moves.size() && tte.num_moves < MAX_TT_MOVES; ++i) {
                int idx = GET_LSB_IDX(moves[i].move_bit);
                if (idx != current_best_move) {
                    tte.packed_moves |= (((uint32_t)idx & 0x3F) << (tte.num_moves * 6));
                    tte.num_moves++;
                }
            }
        }

        return max_score;
    }

    std::pair<int, float> best_move(uint64_t player, uint64_t opponent, int target_depth, bool analyze_mode = false) {
        abort_search = false;
        
        uint64_t hash = calc_hash(player, opponent);
        
        // ★修正: 定石に登録されている場合、最高評価値を持つ手からランダムに1つ選ぶ
        if (opening_book.find(hash) != opening_book.end()) {
            const auto& book_moves = opening_book[hash];
            if (!book_moves.empty()) {
                // 1. 最高評価値を見つける
                float max_score = -1e9f;
                for (const auto& bm : book_moves) {
                    if (bm.second > max_score) {
                        max_score = bm.second;
                    }
                }
                
                // 2. 最高評価値と同等の手(わずかな計算誤差を許容)をすべて集める
                std::vector<int> best_candidates;
                for (const auto& bm : book_moves) {
                    if (bm.second >= max_score - 1e-4f) {
                        best_candidates.push_back(bm.first);
                    }
                }
                
                // 3. 候補の中からランダムに1つ選んで返す
                std::uniform_int_distribution<int> dist(0, (int)best_candidates.size() - 1);
                int selected_move = best_candidates[dist(rng_engine)];
                
                return {selected_move, max_score};
            }
        }

        uint64_t legal = get_valid_moves(player, opponent);
        if (legal == 0ULL) return {-1, 0.0f};

        for (int i = 0; i < 64; ++i) history_table[i] = 0;

        std::vector<RootMove> root_moves;
        while (legal > 0ULL) {
            uint64_t bit = legal & (~legal + 1); 
            root_moves.push_back({bit, GET_LSB_IDX(bit), -1e9f});
            legal ^= bit;
        }

        int global_best_move = root_moves[0].idx;
        float global_best_score = -1e9f;

        for (int d = 1; d <= target_depth; ++d) {
            if (abort_search) break; 

            float alpha = -1e9f;
            float beta = 1e9f;
            float current_depth_best_score = -1e9f;

            for (int i = 0; i < root_moves.size(); ++i) {
                if (abort_search) goto end_search; 

                uint64_t put_bit = root_moves[i].put_bit;
                uint64_t flipped = get_flipped_bb(put_bit, player, opponent);
                uint64_t new_player = player ^ (put_bit | flipped);
                uint64_t new_opponent = opponent ^ flipped;

                float score;
                if (i == 0 || analyze_mode) {
                    score = -nega_scout(new_opponent, new_player, d - 1, false, -beta, -alpha);
                } else {
                    score = -nega_scout(new_opponent, new_player, d - 1, false, -alpha - NULL_WINDOW_EPSILON, -alpha);
                    if (score > alpha && score < beta) {
                        score = -nega_scout(new_opponent, new_player, d - 1, false, -beta, -alpha);
                    }
                }

                root_moves[i].score = score;
                if (score > current_depth_best_score) current_depth_best_score = score;
                if (!analyze_mode && score > alpha) alpha = score;
            }
            
            std::sort(root_moves.begin(), root_moves.end());
            
            global_best_move = root_moves[0].idx;
            global_best_score = root_moves[0].score;
        }

    end_search:
        if (abort_search) {
            return {-1, 0.0f}; 
        }

        last_root_scores.clear();
        for (const auto& rm : root_moves) {
            last_root_scores.push_back({rm.idx, rm.score});
        }

        return {global_best_move, global_best_score};
    }

    std::vector<std::pair<int, float>> get_root_scores() {
        return last_root_scores;
    }
};

PYBIND11_MODULE(engine, m) {
    m.doc() = "C++ Othello Engine with Random Opening Book Selection";

    m.def("get_valid_moves", &get_valid_moves);
    m.def("get_flipped_bb", &get_flipped_bb);

    py::class_<SearchEngine>(m, "SearchEngine")
        .def(py::init<>())
        .def("set_structure", &SearchEngine::set_structure)
        .def("load_weights", &SearchEngine::load_weights)
        .def("add_book_move", &SearchEngine::add_book_move)
        .def("clear_book", &SearchEngine::clear_book)
        .def("stop", &SearchEngine::stop) 
        .def("get_board_hash", &SearchEngine::get_board_hash)
        .def("best_move", &SearchEngine::best_move, 
             py::arg("player"), py::arg("opponent"), 
             py::arg("target_depth"), py::arg("analyze_mode") = false,
             py::call_guard<py::gil_scoped_release>()) 
        .def("get_root_scores", &SearchEngine::get_root_scores)
        .def("clear_tt", &SearchEngine::clear_tt);
}