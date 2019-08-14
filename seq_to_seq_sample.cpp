#include <cassert>
#include <iostream>
#include <fstream>

#include "low_level_func.h" 
#include "seq_to_seq.h" 
#include "Timing.h" 

using namespace std;

#define NUMBER_OF_WRITES 20
#define SCORE_VALUE_INITIALIZER 1.0f
#define DROP_INDEX (-1)
#define STATS 0

int _num_dec_lstm_calls = 0;
#if STATS
static int _short_rewrite = 0;
static int _dropped = 0;
#endif

SeqToSeq::SeqToSeq() 
	: _enc_hidden_layer(nullptr), _enc_lstm(nullptr), _dec_hidden_layer(nullptr), _dec_lstm(nullptr),
          _enc_embedding(nullptr), _enc_hidden_layer_output(nullptr), _enc_lstm_hidden_states(nullptr),
          _enc_lstm_cell_states(nullptr),
          _dec_hidden_layer_output_for_non(nullptr), _dec_hidden_layer_batched_output(nullptr),
          _dec_batched_embedding(nullptr),
          _dec_lstm_hidden_states_prev(nullptr), _dec_lstm_hidden_states_curr(nullptr),
          _dec_lstm_cell_states_prev(nullptr), _dec_lstm_cell_states_curr(nullptr),
          _batched_next_index(nullptr), _batched_scores(nullptr),
          _prev_layer_beam_result(nullptr), _curr_layer_beam_result(nullptr) { }

SeqToSeq::~SeqToSeq() {
    if(_enc_hidden_layer) delete _enc_hidden_layer;
    if(_enc_lstm) delete _enc_lstm;
    if(_enc_embedding) FreeBufferMKL(_enc_embedding);
    if(_enc_hidden_layer_output) FreeBufferMKL(_enc_hidden_layer_output);
    if(_enc_lstm_hidden_states) {
        for(int i=0; i < _enc_lstm_layers; i++) {
            FreeBufferMKL(_enc_lstm_hidden_states[i]);
        }
        free(_enc_lstm_hidden_states);
    }
    if(_enc_lstm_cell_states) {
        for(int i=0; i < _enc_lstm_layers; i++) {
            FreeBufferMKL(_enc_lstm_cell_states[i]);
        }
        free(_enc_lstm_cell_states);
    }

    if(_dec_hidden_layer) delete _dec_hidden_layer;
    if(_dec_lstm) delete _dec_lstm;

    if(_dec_batched_embedding) FreeBufferMKL(_dec_batched_embedding);
    if(_dec_hidden_layer_output_for_non) FreeBufferMKL(_dec_hidden_layer_output_for_non);
    if(_dec_hidden_layer_batched_output) FreeBufferMKL(_dec_hidden_layer_batched_output);

    if(_dec_lstm_hidden_states_prev) {
        assert(_dec_lstm_hidden_states_curr);
        for(int i=0; i < _dec_lstm_layers; i++) {
            FreeBufferMKL(_dec_lstm_hidden_states_prev[i]);
            FreeBufferMKL(_dec_lstm_hidden_states_curr[i]);
        }
        free(_dec_lstm_hidden_states_prev);
        free(_dec_lstm_hidden_states_curr);
    }
    if(_dec_lstm_cell_states_prev) {
        assert(_dec_lstm_cell_states_curr);
        for(int i=0; i < _dec_lstm_layers; i++) {
            FreeBufferMKL(_dec_lstm_cell_states_prev[i]);
            FreeBufferMKL(_dec_lstm_cell_states_curr[i]);
        }
        free(_dec_lstm_cell_states_prev);
        free(_dec_lstm_cell_states_curr);
    }

    if(_prev_layer_beam_result) {
        assert(_curr_layer_beam_result);
        for(int i=0; i < _dec_max_batch_size; i++) {
            free(_prev_layer_beam_result[i].tokens);
            free(_curr_layer_beam_result[i].tokens);
        }
        free(_prev_layer_beam_result);
        free(_curr_layer_beam_result);
    }

    if(_batched_next_index) { free(_batched_next_index); }
    if(_batched_scores) { free(_batched_scores); }
}

void SeqToSeq::ResetEncoderLSTMStates() {
    for(int i=0; i < _enc_lstm_layers; i++) {
        for(int j=0; j < (_enc_hidden_dim * _enc_max_seq_num); j++) {
            _enc_lstm_hidden_states[i][j] = 0.0f;
        }
        for(int j=0; j < _enc_hidden_dim; j++) {
            _enc_lstm_cell_states[i][j] = 0.0f;
        }
    }
    _enc_lstm->SetStates(_enc_lstm_hidden_states, _enc_lstm_cell_states);
}

bool SeqToSeq::Encode(const vector<string> &input, prof_timer_t * timer) {

    const int seq_length = input.size();
    int seq_lengths[1] = { seq_length }; // array of size 1

    // Read Input Word2Vec for encoder
    TIMER_ENTER_STATE(timer, EMBEDDING);
    bool res = _m_emb->Run(input, _enc_embedding);
    TIMER_LEAVE_STATE(timer, EMBEDDING);
    if(res == false) return false;

    TIMER_ENTER_STATE(timer, ENCODING);
    // Run the embedded input through the encoder hidden layer; reinitialize the lstm states
    ResetEncoderLSTMStates();    
#ifndef SMALL_MODEL
    mkl_set_num_threads_local(3);
    _enc_hidden_layer->Compute(_enc_embedding, seq_length, _enc_hidden_layer_output);
    mkl_set_num_threads_local(0);

    _enc_lstm->Compute(_enc_hidden_layer_output, seq_length, 1/*batch*/, nullptr, _enc_lstm_hidden_states, 2);
#else
    _enc_lstm->Compute(_enc_embedding, seq_length, 1/*batch*/, nullptr, _enc_lstm_hidden_states, 2);
#endif
    TIMER_LEAVE_STATE(timer, ENCODING);

#if TIMING
    // This doesn't count the hidden layer 
    int64_t hidden_layer_ops = (int64_t)(_enc_emb_dim * _enc_hidden_dim);
    int64_t stacked_lstm_ops = (int64_t)(_enc_hidden_dim + _enc_hidden_dim) * _enc_hidden_dim * 4 * 2 * _enc_lstm_layers;
    int64_t total_num_ops = (hidden_layer_ops + stacked_lstm_ops) * seq_length;
    TIMER_ADD_TO_OPS(timer, ENCODING, total_num_ops);
#endif
    
    return true;
}

// The root is a little bit special:
// 0. We have already computed the decoder hidden layer output for <NON>
// 1. It uses the hidden and cell states from the encoder LSTM
// 2. We call RunScoring instead of RunScoringOptimized, which sets up the inital small wordset pool
// 3. There is not much filtering except that we check the next index against the Adjective list
int SeqToSeq::DecodeAtBeamSearchRoot(int input_seq_length, prof_timer_t * timer) {

    int seq_lengths[1] = {1};
    const int topK = _c_topK[0];
    int batch_size = 0;

    // Now we initialize decoder lstm with encoders hidden output and cell states 
    int offset = _enc_hidden_dim * (input_seq_length - 1);
    for(int i=0; i < _dec_lstm_layers; i++) {
        memcpy(_dec_lstm_hidden_states_prev[i], _enc_lstm_hidden_states[i] + offset, 
                sizeof(float) * _dec_hidden_dim);
    }
    _dec_lstm->SetStates(_dec_lstm_hidden_states_prev, _enc_lstm_cell_states);

    TIMER_ENTER_STATE(timer, DECODING);
#ifndef SMALL_MODEL
    assert(_dec_hidden_layer_output_for_non);
    // mkl_set_num_threads(3);
    _dec_lstm->Compute(_dec_hidden_layer_output_for_non, 1/*seq_length*/, 1/*batch*/, 
                       seq_lengths, _dec_lstm_hidden_states_curr, 2);
#else
    const float *dec_embedding = _m_emb->ReturnTargetWord2Vec(WS_INDEX_FOR_NON);
    _dec_lstm->Compute((float *)dec_embedding, 1/*seq_length*/, 1/*batch*/, 
                       seq_lengths, _dec_lstm_hidden_states_curr, 2);
#endif
    TIMER_LEAVE_STATE(timer, DECODING);

#if TIMING
    // This doesn't count the hidden layer 
    int64_t hidden_layer_ops = _dec_emb_dim * _dec_hidden_dim;
    int64_t total_num_ops = hidden_layer_ops + (int64_t)(_dec_hidden_dim + _dec_hidden_dim) * _dec_hidden_dim * 4 * 2 * _dec_lstm_layers;
    TIMER_ADD_TO_OPS(timer, ENCODING, total_num_ops);
#endif
    _m_pool->RunScoringParallel(topK, _dec_lstm_hidden_states_curr[_dec_lstm_layers-1], 
                        _batched_next_index, _batched_scores, timer);
#if _DEBUG
    cout << "Root layer results:" << endl;
    for (int b = 0; b < topK; b++) {
        cout << " -- next index = " << _batched_next_index[b] << ", next score = " << _batched_scores[b] << endl;
    }
#endif

    string next_word = "";
    int k; 
    // possibly optimize later
    // Set up the initial LSTM cell states
    for(int i=0; i < topK; i++) {
        _m_emb->TargetDicLookup(_batched_next_index[i], next_word);
        // Maybe need to drop the m_adj later
        if(_batched_next_index[i] == 0) { // drop it here
            _batched_next_index[i] = DROP_INDEX;
#if STATS
            _dropped++;
#endif
        } else {
            _prev_layer_beam_result[batch_size].score = (double) _batched_scores[i];
            _prev_layer_beam_result[batch_size].tokens[0] = _batched_next_index[i];
            _prev_layer_beam_result[batch_size].num_tokens = 1;
            batch_size++;
        }
    }

    for(int i=0; i < _dec_lstm_layers; i++) {
        CopyVectorWithRepeats(_dec_lstm_hidden_states_curr[i], _dec_hidden_dim,   
                              batch_size, _dec_lstm_hidden_states_prev[i]);
    }
    for(int i=0; i < _dec_lstm_layers; i++) {
        CopyVectorWithRepeats(_enc_lstm_cell_states[i], _dec_hidden_dim,
                              batch_size, _dec_lstm_cell_states_prev[i]);
    }
    _dec_lstm->SetStates(_dec_lstm_hidden_states_prev, _dec_lstm_cell_states_prev);

    return batch_size;
}

void SeqToSeq::RunOneLayerBeamSearch(const int batch_size, prof_timer_t * timer) {

    int offset = 0;
    int seq_lengths[1] = {1};
    int i = 0; // used to index _batched_next_index 
    int count = 0; // used to keep track of batch_size
        // since some of the _batched_next_index is marked as DROP_INDEX, we
        // may need to go further on _bathced_next_index before we get to batch_size 

    while(count < batch_size) {
        int next_index = _batched_next_index[i++];
        if( next_index == DROP_INDEX ) { continue; }
        count++;
        const float *dec_embedding = _m_emb->ReturnTargetWord2Vec(next_index);
        assert(dec_embedding != nullptr);
        memcpy(_dec_batched_embedding + offset, dec_embedding, sizeof(float)*_dec_emb_dim);
        offset += _dec_emb_dim;
    }
    TIMER_ENTER_STATE(timer, DECODING);
    // mkl_set_num_threads(3);
#ifndef SMALL_MODEL
    mkl_set_num_threads_local(3);
    _dec_hidden_layer->Compute(_dec_batched_embedding, batch_size, _dec_hidden_layer_batched_output);
    mkl_set_num_threads_local(0);

    // Don't need to SetNumThreads if using Compute
    // _dec_lstm->SetNumThreads(4/*max thread*/, 4/*hidden max thread*/, batch_size);
    // _dec_lstm->ComputeV2(_dec_hidden_layer_batched_output, 1/*seq_length*/, batch_size, 
    //                      seq_lengths, _dec_lstm_hidden_states_curr, 1/*threads*/);
    // mkl_set_num_threads(2);
    _dec_lstm->Compute(_dec_hidden_layer_batched_output, 1/*seq_length*/, batch_size, 
                       nullptr, _dec_lstm_hidden_states_curr, 2);
#else 
    _dec_lstm->Compute(_dec_batched_embedding, 1/*seq_length*/, batch_size, 
                       nullptr, _dec_lstm_hidden_states_curr, 2);
#endif

    TIMER_LEAVE_STATE(timer, DECODING);
#if TIMING
    // This doesn't count the hidden layer 
    // int64_t hidden_layer_ops = _dec_emb_dim * _dec_hidden_dim;
    int64_t hidden_layer_ops = 0;
    int64_t stacked_lstm_ops = (int64_t)(_dec_hidden_dim + _dec_hidden_dim) * _dec_hidden_dim * 4 * 2 * _dec_lstm_layers;
    int64_t total_num_ops = (hidden_layer_ops + stacked_lstm_ops) * batch_size; 
    TIMER_ADD_TO_OPS(timer, DECODING, total_num_ops);
#endif
}

// Input:
// - topK: the number of topK used to process _prev_layer_beam_result
// - batch_size: indicates the valid entires in the _prev_layer_beam_result.
// - _prev_layer_beam_result: the result from the previous batch of beam search 
// - _batched_next_index: next index to explore with dropout marked; for each element in
//       _prev_layer_beam_result, there are topK next index in _batched_next_index
// - _batched_scores: the corresponding scores for the next indices in _batched_next_index
// - to_keep: how many of the next indices we are going to keep; size: batch_size, one per 
//      element in _prev_layer_beam_result
//
// Output: based on the inputs above, 
// - set up the _curr_layer_beam_result
// - set up the hidden and cell states for the next LSTM decoder computation
// - return value is the new batch size for the next layer of beam search
int SeqToSeq::SetStatesForNextLayerBeamSearch(const int topK, const int batch_size, const int *to_keep) {

    // REMOVE check variable later
    int k1 = 0, k2 = 0;
    for(int i=0; i < batch_size; i++) {
        int check = 0;
        const BeamSearchData *bsd = &(_prev_layer_beam_result[i]);
        for(int j=0; j < topK; j++) {
            if(_batched_next_index[k1] != DROP_INDEX) {
                _curr_layer_beam_result[k2].score = bsd->score * _batched_scores[k1];
                assert(bsd->num_tokens < _c_topK.size());
                memcpy(_curr_layer_beam_result[k2].tokens, bsd->tokens, bsd->num_tokens*sizeof(int));
                _curr_layer_beam_result[k2].tokens[bsd->num_tokens] = _batched_next_index[k1];
                _curr_layer_beam_result[k2].num_tokens = bsd->num_tokens + 1;
                k2++;
                check++;
            }
            k1++;
        }
        assert(check == to_keep[i]);
    }
    assert(k2 <= (topK * batch_size));
    SeqToSeq::SwapPointers<BeamSearchData *>(&_prev_layer_beam_result, &_curr_layer_beam_result);

    int check = 0;
    // LSTM Compute reads previous hidden states and output to curr_hidden_states, so the 
    // data from previous layer is in _dec_lstm_hidden_states_curr
    for(int i=0; i < _dec_lstm_layers; i++) {
        check = 0;
        float *src = _dec_lstm_hidden_states_curr[i];
        float *dst = _dec_lstm_hidden_states_prev[i];
        for(int j=0; j < batch_size; j++) {
            /* params: src, length, repeat, dst */
            CopyVectorWithRepeats(src, _dec_hidden_dim, to_keep[j], dst);
            src = src + _dec_hidden_dim;
            dst = dst + _dec_hidden_dim * to_keep[j];
            check += to_keep[j];
        }

        assert(check == k2);
    }
    if(check != k2) {
        cout << "Here.\n";
    }
    assert(check == k2);

    // LSTM Compute updates cell states in place, so the data from previous layer
    // is in _dec_lstm_cell_states_prev; so we should swap them first
    SeqToSeq::SwapPointers<float **>(&_dec_lstm_cell_states_prev, &_dec_lstm_cell_states_curr);
    for(int i=0; i < _dec_lstm_layers; i++) {
        float *src = _dec_lstm_cell_states_curr[i];
        float *dst = _dec_lstm_cell_states_prev[i];
        for(int j=0; j < batch_size; j++) {
            /* params: src, length, repeat, dst */
            CopyVectorWithRepeats(src, _dec_hidden_dim, to_keep[j], dst);
            src = src + _dec_hidden_dim;
            dst = dst + _dec_hidden_dim * to_keep[j];
        }
    }

    _dec_lstm->SetStates(_dec_lstm_hidden_states_prev, _dec_lstm_cell_states_prev);

    return k2;
}

void SeqToSeq::ResetBeamSearchResult(int batch_size) {
    for(int i=0; i < batch_size; i++) {
        _prev_layer_beam_result[i].score = SCORE_VALUE_INITIALIZER;
        _prev_layer_beam_result[i].num_tokens = 0;
    }
    for(int i=0; i < batch_size; i++) {
        _curr_layer_beam_result[i].score = SCORE_VALUE_INITIALIZER;
        _curr_layer_beam_result[i].num_tokens = 0;
    }
    /*
    for(int i=batch_size; i < _dec_max_batch_size; i++) {
        assert(_prev_layer_beam_result[i].score == SCORE_VALUE_INITIALIZER);
        assert(_curr_layer_beam_result[i].score == SCORE_VALUE_INITIALIZER);
        assert(_prev_layer_beam_result[i].num_tokens == 0);
        assert(_curr_layer_beam_result[i].num_tokens == 0);
    }*/
}

bool SeqToSeq::Run(const vector<string> &input, vector<string>& output, prof_timer_t * timer) {

#if STATS
    int batch_ave = 1; // for the Root call
    int count = 0;
    _dropped = 0;
    _short_rewrite = 0;
#endif

    bool res = Encode(input, timer);
    if(res == false) return false;

    TIMER_ENTER_STATE(timer, REWRITE);
    int batch_size = DecodeAtBeamSearchRoot(input.size(), timer); // generate the first _c_topK[0] top tokens 
    int max_batch_size = batch_size;

#if STATS
    count++;
    cout << "batch: " << batch_size << " ";
#endif

    // Each iteration is effectively a layer of Beam Search BFS (in level i)
    for(int i = 1; (i < _c_topK.size()) && (batch_size > 0); i++) {
        const int topK = _c_topK[i];
        int to_keep[batch_size];

#if STATS
        batch_ave += batch_size;
        count++;
        cout << "batch: " << batch_size << " ";
#endif
        RunOneLayerBeamSearch(batch_size, timer);
        _m_pool->RunScoringOptimizedParallel2(topK, batch_size, _dec_lstm_hidden_states_curr[_dec_lstm_layers-1],
                                   _batched_next_index, _batched_scores, 8, timer);
#if _DEBUG
        cout << "Beam search layer " << i << " results:" << endl;
        for (int b = 0; b < batch_size * topK; b++) {
            cout << "Batch number " << b << ": Index = " << _batched_next_index[b] << ": Score = " << _batched_scores[b] << endl;
            
        }
#endif
        CheckForTermination(topK, batch_size, to_keep);
        // Even when topK == 1, still need to SetState because of the drop outs
        batch_size = SetStatesForNextLayerBeamSearch(topK, batch_size, to_keep);
        if(max_batch_size < batch_size) max_batch_size = batch_size;
    }

    // everything still in the batch gets added to rewrite
    for(int i=0; i < batch_size; i++) {
        const BeamSearchData *bsd = &(_prev_layer_beam_result[i]);
        AddToRewrite(bsd);
    }
    ReturnFinalRewrites(output);
    _rewrite_scores.clear();
    TIMER_LEAVE_STATE(timer, REWRITE);

    // set up for next query: clear out the beam search result
    ResetBeamSearchResult(max_batch_size);
#if STATS
    batch_ave += batch_size;
    _num_dec_lstm_calls += batch_ave;
    cout << "\nNum of LSTM calls: " << batch_ave << endl << flush;
    cout << "Average of batch size: " << batch_ave / count << ", out of " << count << " batch" << endl << flush;

    cout << "Number of paraphrases: " << batch_size << endl << flush;
    cout << "Number of short paraphrases: " << _short_rewrite << endl << flush;
    cout << "Number of dropped: " << _dropped << endl << flush;
#endif
    return true; // change to void later 
} 

string SeqToSeq::ConvertHashToString(const BeamSearchData *bsd) {
    string new_rewrite = "";
    for(int i=0; i < bsd->num_tokens; i++) {
        string newKey = "";
        _m_emb->TargetDicLookup(bsd->tokens[i], newKey);
        new_rewrite = new_rewrite + " " + newKey;
    }

    return new_rewrite; // this should make a copy
}

template<typename T> 
void SeqToSeq::SwapPointers(T *p1, T *p2) {
    T tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
}

void SeqToSeq::AddToRewrite(const BeamSearchData *bsd) {
    assert(bsd->num_tokens > 0);
    double power_val = (double) 1.0 / (double)(bsd->num_tokens);
    double output = (double)pow(bsd->score, power_val);
    _rewrite_scores[ConvertHashToString(bsd)] = output;
}

// If there isn't any rewrite_score, return true with return empty output;
// Otherwise, 
//      if there are less rewrite than the topK we want, return those; 
//      otherwise, return topK
//  rewrite later; inefficient.
void SeqToSeq::ReturnFinalRewrites(vector<string>& output) {
    vector<double> scores;
    int counter = 0;
    for (map<string, double>::iterator it = _rewrite_scores.begin(); 
            it != _rewrite_scores.end(); ++it) {
        scores.push_back(it->second);
        counter++;
    }
    std::sort(scores.rbegin(), scores.rend());

    if (counter == 0) {
        assert(output.empty()); 
        return;
    }

    double threshold = 0;
    if(_num_of_rewrites > counter) {
        threshold = scores[counter - 1];
    } else {
        threshold = scores[_num_of_rewrites - 1];
    }
    for(map<string, double>::iterator it = _rewrite_scores.begin(); 
                it != _rewrite_scores.end(); ++it) {
        if(it->second >= threshold) {
            output.push_back(it->first);
        }
    }
    return;
}

// Check if the next index is a duplicate, either in the true sense (i.e., same index value)
// or share the same stem (though now we seem to be checking only words > 3 letters and 
// consider them as sharing stem if the first 3 letters are the same).
bool SeqToSeq::IsDuplicateWithStemming(const int next_index, const BeamSearchData *bsd) {
#define CUTOFF_LENGTH 3
    string input_str = "";
    _m_emb->TargetDicLookup(next_index, input_str);

    for(int i=0; i < bsd->num_tokens; i++) {
        string other_str = "";
        if(next_index == bsd->tokens[i]) {
            return true; 
        }
        if (input_str.length() < CUTOFF_LENGTH) { continue; }
        _m_emb->TargetDicLookup(bsd->tokens[i], other_str); 
        if(other_str.length() < CUTOFF_LENGTH) { continue; }
        if(input_str.substr(0, CUTOFF_LENGTH-1) == other_str.substr(0, CUTOFF_LENGTH-1)) {
            return true; 
        }
    }
    return false;
}

// Input:
// - topK: the number of topK used to process _prev_layer_beam_result 
// - batch_size: indicates the valid entires in the _prev_layer_beam_result.
// - _prev_layer_beam_result: the result from the previous batch of beam search 
// - _batched_next_index: next index to explore; for each element in _prev_layer_beam_result,
//      there are topK next index in _batched_next_index 
// - _batched_scores: the corresponding scores for the next indices in _batched_next_index
//
// Output:
// - to_keep: how many of the next indices we are going to keep; size: batch_size, one per 
//      element in _prev_layer_beam_result
// This function walks through the _bathced_next_index and _batched_scores to determine
// which next_index to explore.  For each element in the _prev_layer_beam_result (which corresponds
// to a path in the beam search), we look through _batched_next_index / _batched_scores to determine 
// if we want to explore that index next.  If we decide to drop an element at index i in
// _batched_next_index, we mark it '-1' (since we have no negative index for the word set).  
// Update to_keep as we go. 
void SeqToSeq::CheckForTermination(const int topK, const int batch_size, int *to_keep) {

    int k = 0;

    for(int i=0; i < batch_size; i++) {
        to_keep[i] = 0;
        const BeamSearchData *bsd = &(_prev_layer_beam_result[i]);
        for(int j=0; j < topK; j++) {
            int next_index = _batched_next_index[k];
            assert(next_index != DROP_INDEX);
            if( next_index == 0 || IsDuplicateWithStemming(next_index, bsd) ) {
                _batched_next_index[k] = DROP_INDEX;
                AddToRewrite(bsd);
#if STATS
                _short_rewrite++;
#endif
            } else {
                to_keep[i]++;
            }
            k++;
        }
    }
}


