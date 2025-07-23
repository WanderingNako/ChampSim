#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <utility>
#include <vector>

#include "cache.h"
#include "msl/bits.h"

#define SAT_INC(x,max)  (x < max) ? x + 1 : x
#define SAT_DEC(x)      (x > 0)   ? x - 1 : x

namespace
{
constexpr int maxRRPV = 3;
constexpr std::size_t SHCT_SIZE = 16384;
constexpr unsigned SHCT_PRIME = 16381;
constexpr std::size_t SAMPLER_SET = (256 * NUM_CPUS);
constexpr unsigned SHCT_MAX = 7;

// sampler structure
class SAMPLER_class
{
public:
  bool reuse        = false;
  uint32_t signature = 0;
};

// sampler
std::map<CACHE*, std::vector<std::size_t>> rand_sets;
std::map<CACHE*, std::vector<SAMPLER_class>> sampler;
std::map<CACHE*, std::vector<int>> rrpv_values;
std::map<CACHE*, std::vector<bool>> is_prefetch;

// prediction table structure
std::map<std::pair<CACHE*, std::size_t>, std::array<unsigned, SHCT_SIZE>> SHCT;
} // namespace

// initialize replacement state
void CACHE::initialize_replacement()
{
  // randomly selected sampler sets
  std::size_t rand_seed = 1103515245 + 12345;
  ;
  for (std::size_t i = 0; i < ::SAMPLER_SET; i++) {
    std::size_t val = (rand_seed / 65536) % NUM_SET;
    std::vector<std::size_t>::iterator loc = std::lower_bound(std::begin(::rand_sets[this]), std::end(::rand_sets[this]), val);

    while (loc != std::end(::rand_sets[this]) && *loc == val) {
      rand_seed = rand_seed * 1103515245 + 12345;
      val = (rand_seed / 65536) % NUM_SET;
      loc = std::lower_bound(std::begin(::rand_sets[this]), std::end(::rand_sets[this]), val);
    }

    ::rand_sets[this].insert(loc, val);
  }

  sampler.emplace(this, ::SAMPLER_SET * NUM_WAY);

  ::rrpv_values[this] = std::vector<int>(NUM_SET * NUM_WAY, ::maxRRPV);
  ::is_prefetch[this] = std::vector<bool>(NUM_SET * NUM_WAY, false);
}

// find replacement victim
uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type)
{
  // look for the maxRRPV line
  auto begin = std::next(std::begin(::rrpv_values[this]), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);
  auto victim = std::find(begin, end, ::maxRRPV);
  while (victim == end) {
    for (auto it = begin; it != end; ++it)
      ++(*it);

    victim = std::find(begin, end, ::maxRRPV);
  }

  assert(begin <= victim);
  return static_cast<uint32_t>(std::distance(begin, victim)); // cast pretected by prior assert
}

// called on every cache hit and cache fill
void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{
  auto s_idx     = std::find(std::begin(::rand_sets[this]), std::end(::rand_sets[this]), set);
  bool is_sample = (s_idx != std::end(::rand_sets[this]));
  uint32_t loc   = is_sample ? std::distance(std::begin(::rand_sets[this]), s_idx) * NUM_WAY + way : -1;
  uint32_t sig   = is_sample ? ::sampler[this][loc].signature : -1;
  auto fill_cpu  = std::make_pair(this, triggering_cpu);
  // handle hit
  if (hit && access_type{type} != access_type::WRITE) {
    // SHiP++ : Prefetch-Aware RRPV Updates(scenarios 2)
    if (access_type{type} == access_type::PREFETCH &&  ::is_prefetch[this][set * NUM_WAY + way]) {
        // 5% chance to update SHCT
        if (is_sample && (rand()%100 <5)){
            ::SHCT[fill_cpu][sig]      = SAT_INC(::SHCT[fill_cpu][sig], ::SHCT_MAX);
            ::sampler[this][loc].reuse = true;
        }
    }
    else {
        ::rrpv_values[this][set * NUM_WAY + way] = 0;
        // SHiP++ : Prefetch-Aware RRPV Updates(scenarios 1)
        if (::is_prefetch[this][set * NUM_WAY + way]){
            ::rrpv_values[this][set * NUM_WAY + way] = ::maxRRPV;
            ::is_prefetch[this][set * NUM_WAY + way] = false;
        }
        // SHiP++ : Improved SHCT Training
        if (is_sample && (::sampler[this][loc].reuse == false)){
            ::SHCT[fill_cpu][sig]      = SAT_INC(::SHCT[fill_cpu][sig], ::SHCT_MAX);
            ::sampler[this][loc].reuse = true;
        }
    }
    return;
  }
  //--- All of the below is done only on misses -------
  // remember signature of what is being inserted
  // SHiP : Prefetch-Aware SHCT Training
  uint64_t use_PC  = (access_type{type} == access_type::PREFETCH) ? ((ip << 1) + 1) : (ip << 1);
  uint32_t new_sig = use_PC % ::SHCT_PRIME;

  if (is_sample){
    // update signature based on what is getting evicted
    if (::sampler[this][loc].reuse == false){
        ::SHCT[fill_cpu][sig] = SAT_DEC(::SHCT[fill_cpu][sig]);
    }
    ::sampler[this][loc].reuse     = false;
    ::sampler[this][loc].signature = new_sig;
  }

  ::is_prefetch[this][set * NUM_WAY + way] = access_type{type} == access_type::PREFETCH;

  // Now determine the insertion prediciton
  // Scan + WriteBack
  if ((access_type{type} == access_type::WRITE) || (::SHCT[fill_cpu][new_sig] == 0)){
    ::rrpv_values[this][set * NUM_WAY + way] = ::maxRRPV;
  }
  // High-confidence
  else if (::SHCT[fill_cpu][new_sig] == ::SHCT_MAX){
    ::rrpv_values[this][set * NUM_WAY + way] = 0;
  }
  else {
    ::rrpv_values[this][set * NUM_WAY + way] = ::maxRRPV - 1;
  }
  
}

// use this function to print out your own stats at the end of simulation
void CACHE::replacement_final_stats() {}