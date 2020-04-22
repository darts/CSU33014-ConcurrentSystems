//
// CSU33014 Summer 2020 Additional Assignment
// Part A of a two-part assignment
//

// Please examine version each of the following routines with names
// starting partA. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics. Where is cannot be vectorized...

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".

// The following was used as a source of information on instructions and their
// cost: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#

#include <immintrin.h>
#include <stdio.h>

#include "csu33014-annual-partA-code.h"

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void partA_routine0(float *restrict a, float *restrict b, float *restrict c) {
  for (int i = 0; i < 1024; i++) {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void partA_vectorized0(float *restrict a, float *restrict b,
                       float *restrict c) {
  __m128 a4, b4, c4;
  for (int i = 0; i < 1024; i = i + 4) {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float partA_routine1(float *restrict a, float *restrict b, int size) {
  float sum = 0.0;

  for (int i = 0; i < size; i++) {
    sum = sum + a[i] * b[i];
  }
  return sum;
}

// insert vectorized code for routine1 here
float partA_vectorized1(float *restrict a, float *restrict b, int size) {

  // This is the first routine I wrote. It occasionally presents floating point
  // errors:
  // float sum = 0.0;
  // float sum_arr[4];
  // __m128 vec_sum = _mm_setzero_ps();
  // int v;
  // for (v = 0; v < size - 3; v += 4) {
  //   // {a[v], a[v+1], a[v+2], a[v+3]}
  //   __m128 a_vec = _mm_loadu_ps(&a[v]);

  //   // {b[v], b[v+1], b[v+2], b[v+3]}
  //   __m128 b_vec = _mm_loadu_ps(&b[v]);

  //   // add to sum
  //   vec_sum = _mm_add_ps(vec_sum, _mm_mul_ps(a_vec, b_vec));
  // }
  // _mm_storeu_ps(sum_arr, vec_sum);
  // sum += sum_arr[0];
  // sum += sum_arr[1];
  // sum += sum_arr[2];
  // sum += sum_arr[3];

  // this solution solves the floating point inaccuracies (although it may be
  // marginally slower)
  float sum = 0.0;
  float sum_arr[4];
  int v;
  for (v = 0; v < size - 3; v += 4) {
    // {a[v], a[v+1], a[v+2], a[v+3]}
    __m128 a_vec = _mm_loadu_ps(&a[v]);

    // {b[v], b[v+1], b[v+2], b[v+3]}
    __m128 b_vec = _mm_loadu_ps(&b[v]);

    // add to sum
    a_vec = _mm_mul_ps(a_vec, b_vec);

    // store and add to sum
    _mm_storeu_ps(sum_arr, a_vec);
    sum += sum_arr[0];
    sum += sum_arr[1];
    sum += sum_arr[2];
    sum += sum_arr[3];
  }

  for (; v < size; v++) {
    sum = sum + a[v] * b[v];
  }

  return sum;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void partA_routine2(float *restrict a, float *restrict b, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = 1 - (1.0 / (b[i] + 1.0));
  }
}

// in the following, size can have any positive value
void partA_vectorized2(float *restrict a, float *restrict b, int size) {

  __m128 one_mask = _mm_set1_ps((float)1);

  int v;
  for (v = 0; v < size - 3; v += 4) {

    // { b[v], b[v+1], b[v+2], b[v+3] }
    __m128 b_load = _mm_loadu_ps(&b[v]);

    // { b[v]+1, b[v+1]+1, b[v+2]+1, b[v+3]+1 }
    b_load = _mm_add_ps(b_load, one_mask);

    // Originally used this:
    // { 1/(b[v]+1), 1/(b[v+1]+1), 1/(b[v+2]+1), 1/(b[v+3]+1) }
    // b_load = _mm_div_ps(one_mask, b_load);

    // Found this in the Intel docs to replace the above.
    // It is substantially faster and with much lower latency.
    b_load = _mm_rcp_ps(b_load);

    // { 1-(1/(b[v]+1)), 1-(1/(b[v+1]+1)), 1-(1/(b[v+2]+1)), 1-(1/(b[v+3]+1)) }
    b_load = _mm_sub_ps(one_mask, b_load);

    // store result
    _mm_storeu_ps(&a[v], b_load);
  }

  for (; v < size; v++) {
    a[v] = 1 - (1.0 / (b[v] + 1.0));
  }
}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void partA_routine3(float *restrict a, float *restrict b, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i] < 0.0) {
      a[i] = b[i];
    }
  }
}

// in the following, size can have any positive value
void partA_vectorized3(float *restrict a, float *restrict b, int size) {
  // replace the following code with vectorized code
  int v;
  __m128 zero_vec = _mm_setzero_ps();

  // For multiples of 4
  for (v = 0; v < size - 3; v += 4) {
    // {a[v], a[v+1], a[v+2], a[v+3]}
    __m128 a_vec = _mm_loadu_ps(&a[v]);

    // {a[v] < 0, a[v+1] < 0, a[v+2] < 0, a[v+3] < 0}
    // Note the value for true is 0xFFFFFFFF (true) or 0x0 (false)
    __m128 cmp_vec = _mm_cmplt_ps(a_vec, zero_vec);

    // {b[v], b[v+1], b[v+2], b[v+3]}
    __m128 b_vec = _mm_loadu_ps(&b[v]);

    // AND b_vec and cmp_vec
    // Only where a[v_i] < 0, remains
    b_vec = _mm_and_ps(b_vec, cmp_vec);

    // cmp_vec = NOT cmp_vec
    // AND a_vec and cmp_vec
    // Only where a[v_i] >= 0, remains
    a_vec = _mm_andnot_ps(cmp_vec, a_vec);

    // OR the two vectors for a complete set
    // Store them in the array
    _mm_storeu_ps(&a[v], _mm_or_ps(a_vec, b_vec));
  }

  // Remaining multiples
  for (int i = 0; i < size; i++) {
    if (a[i] < 0.0) {
      a[i] = b[i];
    }
  }
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void partA_routine4(float *restrict a, float *restrict b, float *restrict c) {
  for (int i = 0; i < 2048; i = i + 2) {
    a[i] = b[i] * c[i] - b[i + 1] * c[i + 1];
    a[i + 1] = b[i] * c[i + 1] + b[i + 1] * c[i];
  }
}

#define CREATE_IMM8(a, b, c, d) ((d << 6) | (c << 4) | (b << 2) | (a << 0))
void partA_vectorized4(float *restrict a, float *restrict b,
                       float *restrict c) {
  for (int i = 0; i < 2048; i += 4) {
    // Original method:
    //***************************************************************************
    // __m128 b_i_vec = _mm_loadu_ps(&b[i]); // load {b[i], b[i+1], b[i+2],
    // b[i+3]}
    // __m128 c_i_vec = _mm_loadu_ps(&c[i]); // load {c[i], c[i+1], c[i+2],
    // c[i+3]}

    // //{( b[i]*c[i] ), ( b[i+1]*c[i+1] ), ( b[i+2]*c[i+2] ), ( b[i+3]*c[i+3]
    // )}
    // __m128 a_i_vec = _mm_mul_ps(b_i_vec, c_i_vec);

    // //{( b[i]*c[i] ) - ( b[i+1]*c[i+1] ), ( b[i+2]*c[i+2] ) - (
    // b[i+3]*c[i+3]),
    // // ( b[i]*c[i] ) - ( b[i+1]*c[i+1] ), ( b[i+2]*c[i+2] ) - (
    // b[i+3]*c[i+3])} a_i_vec = _mm_hsub_ps(a_i_vec, a_i_vec);

    // // {b[i+1], b[i], b[i+3], b[i+2]}
    // b_i_vec = _mm_shuffle_ps(b_i_vec, b_i_vec, CREATE_IMM8(1, 0, 3, 2));

    // //{( b[i+1]*c[i] ), ( b[i]*c[i+1] ), ( b[i+3]*c[i+2] ), ( b[i+2]*c[i+3])}
    // __m128 a_i_plus_vec = _mm_mul_ps(b_i_vec, c_i_vec);

    // //{( b[i+1]*c[i] ) + ( b[i]*c[i+1] ), ( b[i+3]*c[i+2] ) + ( b[i+2]*c[i+3]
    // ),
    // // ( b[i+1]*c[i] ) + ( b[i]*c[i+1] ), ( b[i+3]*c[i+2] ) + ( b[i+2]*c[i+3]
    // )} a_i_plus_vec = _mm_hadd_ps(a_i_plus_vec, a_i_plus_vec);

    // //{( b[i]*c[i] ) - ( b[i+1]*c[i+1] ), ( b[i+2]*c[i+2] ) - ( b[i+3]*c[i+3]
    // ),
    // // ( b[i+1]*c[i] ) + ( b[i]*c[i+1] ), ( b[i+3]*c[i+2] ) + ( b[i+2]*c[i+3]
    // )} a_i_plus_vec =
    //     _mm_shuffle_ps(a_i_vec, a_i_plus_vec, CREATE_IMM8(0, 1, 0, 1));

    // //{( b[i]*c[i] )-( b[i+1]*c[i+1] ), ( b[i+1]*c[i] )+( b[i]*c[i+1] )
    // // ( b[i+2]*c[i+2] )-( b[i+3]*c[i+3] ), ( b[i+3]*c[i+2] )+( b[i+2]*c[i+3]
    // )} a_i_plus_vec =
    //     _mm_shuffle_ps(a_i_plus_vec, a_i_plus_vec, CREATE_IMM8(0, 2, 1, 3));

    // _mm_storeu_ps(&a[i], a_i_plus_vec); // store result

    // Cost, calculated from Intel docs for Skylake processors:
    /*
    loadu:   latency(0), CPI(0.5) -> x2 latency(0), CPI(1)
    mul:     latency(4), CPI(0.5) -> x2 latency(8), CPI(1)
    hsub:    latency(6), CPI(2)   -> x1 latency(6), CPI(2)
    shuffle: latency(1), CPI(1)   -> x3 latency(3), CPI(3)
    hadd:    latency(6), CPI(2)   -> x1 latency(6), CPI(2)
    storeu:  latency(0), CPI(1)   -> x1 latency(0), CPI(1)

    Total: latency(23), CPI(10)
    */
    //***************************************************************************
    // we can do better by avoiding hadd and hsub. The following code makes
    // better use of the calculations. Some of the operations that were
    // duplicated previously are now not. New method:
    __m128 b_i_vec = _mm_loadu_ps(&b[i]); // load {b[i], b[i+1], b[i+2], b[i+3]}
    __m128 c_i_vec = _mm_loadu_ps(&c[i]); // load {c[i], c[i+1], c[i+2], c[i+3]}

    // { b[i], b[i], b[i+2], b[i+2]}
    __m128 b_select = _mm_shuffle_ps(b_i_vec, b_i_vec, CREATE_IMM8(0, 0, 2, 2));

    // { b[i]+c[i], b[i]*c[i+1], b[i+2]*c[i+2], b[i+2]*c[i+3}
    __m128 i_final = _mm_mul_ps(b_select, c_i_vec);

    // { c[i+1], c[i], c[i+3], c[i+2] }
    __m128 c_select = _mm_shuffle_ps(c_i_vec, c_i_vec, CREATE_IMM8(1, 0, 3, 2));

    // { b[i+1], b[i+1], b[i+3], b[i+3]}
    b_select = _mm_shuffle_ps(b_i_vec, b_i_vec, CREATE_IMM8(1, 1, 3, 3));

    // { c[i+1]*b[i+1], c[i]+b[i+1], c[i+3]*b[i+3], c[i+2]*b[i+3] }
    __m128 i_plus_final = _mm_mul_ps(c_select, b_select);

    //{( b[i]*c[i] )-( b[i+1]*c[i+1] ), ( b[i+1]*c[i] )+( b[i]*c[i+1] )
    // ( b[i+2]*c[i+2] )-( b[i+3]*c[i+3] ), ( b[i+3]*c[i+2] )+( b[i+2]*c[i+3]
    i_final = _mm_addsub_ps(i_final, i_plus_final);

    // Store result
    _mm_storeu_ps(&a[i], i_final);

    // Cost, calculated from Intel docs for Skylake processors:
    /*
    loadu:   latency(0), CPI(0.5) -> x2 latency(0), CPI(1)
    mul:     latency(4), CPI(0.5) -> x2 latency(8), CPI(1)
    shuffle: latency(1), CPI(1)   -> x3 latency(3), CPI(3)
    addsub:  latency(4), CPI(0.5) -> x1 latency(4), CPI(0.5)
    storeu:  latency(0), CPI(1)   -> x1 latency(0), CPI(1)

    Total: latency(15), CPI(6.5)
    */
  }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
void partA_routine5(unsigned char *restrict a, unsigned char *restrict b,
                    int size) {
  for (int i = 0; i < size; i++) {
    a[i] = b[i];
  }
}

void partA_vectorized5(unsigned char *restrict a, unsigned char *restrict b,
                       int size) {
  int v;
  for (v = 0; v < size - 15; v += 16) {
    // load 16 chars
    __m128i b_vect = _mm_loadu_si128((__m128i *)&b[v]);

    // store 16 chars
    _mm_storeu_si128((__m128i *)&a[v], b_vect);
  }

  // parse sets of 4
  int *a_int = (int *)a;
  int *b_int = (int *)b;
  for (int i = v / 4; i < (size - 3) / 4; i++, v += 4) {
    a_int[i] = b_int[i];
  }

  // remaining multiples of 1
  for (; v < size; v++) {
    a[v] = b[v];
  }
}

/********************* routine 6 ***********************/

void partA_routine6(float *restrict a, float *restrict b, float *restrict c) {
  a[0] = 0.0;
  for (int i = 1; i < 1023; i++) {
    float sum = 0.0;
    for (int j = 0; j < 3; j++) {
      sum = sum + b[i + j - 1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void partA_vectorized6(float *restrict a, float *restrict b,
                       float *restrict c) {
  a[0] = 0.0;
  float tmp_sum[4];

  // This is constant
  // { c[0], c[1], c[2], c[3]}
  __m128 c_vect = _mm_loadu_ps(c);
  for (int i = 1; i < 1023; i++) {
    // {b[i-1],b[i],b[i+1],b[i+2]}
    __m128 b_vect = _mm_loadu_ps(&b[i - 1]);

    // {b[i-1]*c[0],b[i]*c[1],b[i+1]*c[2],b[i+2]*c[3]}
    b_vect = _mm_mul_ps(b_vect, c_vect);

    // store result
    _mm_storeu_ps(tmp_sum, b_vect);

    // prevent some floating point errors
    a[i] = tmp_sum[0];
    a[i] += tmp_sum[1];
    a[i] += tmp_sum[2];
  }

  a[1023] = 0.0;
}
