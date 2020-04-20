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
  // replace the following code with vectorized code
  float sum = 0.0;
  float sum_arr[4];
  __m128 vec_sum = _mm_setzero_ps();
  int v;
  for (v = 0; v < size - 3; v += 4) {
    __m128 a_vec = _mm_loadu_ps(&a[v]);
    __m128 b_vec = _mm_loadu_ps(&b[v]);
    vec_sum = _mm_add_ps(vec_sum, _mm_mul_ps(a_vec, b_vec));
  }
  _mm_storeu_ps(sum_arr, vec_sum);
  sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

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
  // replace the following code with vectorized code
  __m128 one_mask = _mm_set1_ps((float)1);

  int v;
  for (v = 0; v < size - 3; v += 4) {
    __m128 b_load = _mm_loadu_ps(&b[v]);
    b_load = _mm_add_ps(b_load, one_mask);
    b_load = _mm_div_ps(one_mask, b_load);
    b_load = _mm_sub_ps(one_mask, b_load);
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

  for (v = 0; v < size - 3; v += 4) {
    __m128 a_vec = _mm_loadu_ps(&a[v]);
    __m128 cmp_vec = _mm_cmplt_ps(a_vec, zero_vec);
    __m128 b_vec = _mm_loadu_ps(&b[v]);
    b_vec = _mm_and_ps(b_vec, cmp_vec);
    a_vec = _mm_andnot_ps(cmp_vec, a_vec);

    _mm_storeu_ps(&a[v], _mm_or_ps(a_vec, b_vec));
  }

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

void partA_vectorized4(float *restrict a, float *restrict b,
                       float *restrict c) {
  // replace the following code with vectorized code
  for (int i = 0; i < 2048; i = i + 2) {
    a[i] = b[i] * c[i] - b[i + 1] * c[i + 1];
    a[i + 1] = b[i] * c[i + 1] + b[i + 1] * c[i];
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
  // replace the following code with vectorized code
  for (int i = 0; i < size; i++) {
    a[i] = b[i];
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
  // replace the following code with vectorized code
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
