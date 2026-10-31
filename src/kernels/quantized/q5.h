#pragma once

#include <cstddef>
#include <cstdint>

namespace edgellm::quantized {

// Byte layouts of the pinned GGML Q5_0 / Q8_0 blocks (32 elements).
// half_scale stores IEEE binary16 bits. Q5 low nibbles address elements 0..15;
// high nibbles address 16..31. qh bit i is the fifth bit of element i.
// Signed Q5 integer = (low four bits | fifth bit << 4) - 16.
struct Q5Block { std::uint16_t half_scale; std::uint8_t high[4]; std::uint8_t low[16]; };
struct Q8Block { std::uint16_t half_scale; std::int8_t values[32]; };
static_assert(sizeof(Q5Block) == 22 && offsetof(Q5Block, high) == 2 && offsetof(Q5Block, low) == 6);
static_assert(sizeof(Q8Block) == 34 && offsetof(Q8Block, values) == 2);

float half_to_float(std::uint16_t bits);
void decode_q5(const Q5Block& block, float* output);  // output: 32 floats
void decode_q8(const Q8Block& block, float* output);  // output: 32 floats

// Correctness reference, deliberately scalar; NOT an optimized candidate.
// Integer block dots accumulate exactly in int32; block scales/results use FP32.
// n must be divisible by 32. n == 0 returns zero without reading pointers.
// Nonzero n needs valid block arrays, no stronger alignment than these structs.
// Finite scales/non-overflowing results are the caller's numerical contract.
// Invalid width/null input throws before reads. No allocation or threading.
float dot_q5_q8_reference(const Q5Block* weights, const Q8Block* activation, std::size_t n);

}  // namespace edgellm::quantized
