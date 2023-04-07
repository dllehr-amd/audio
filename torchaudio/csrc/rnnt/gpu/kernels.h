#pragma once

#include <cassert>

#ifdef __HIP_PLATFORM_AMD__
#include <torchaudio/csrc/rnnt/hip/kernel_utils.h>
#include <torchaudio/csrc/rnnt/hip/math_hip.cuh>
#else
#include <torchaudio/csrc/rnnt/gpu/kernel_utils.h>
#include <torchaudio/csrc/rnnt/gpu/math.cuh>
#endif

#include <stdio.h>
#define DEVICE_INLINE __device__ inline __attribute__((always_inline))
namespace torchaudio {
namespace rnnt {

struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(at::Half* p) {
    p[0] = __low2half(a);
    p[1] = __high2half(a);
    p[2] = __low2half(b);
    p[3] = __high2half(b);
  }
};

template <typename T>
struct Vec4T {};

template <>
struct Vec4T<float> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    //printf("We are in the default constructor\n");
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }
  DEVICE_INLINE Vec4T(const float* p) {
    //printf("We are in the 1 argument constructor\n");

    acc = *((const float4*)p);
  }
  DEVICE_INLINE Vec4T(const float* p, float c) {
    //printf("We are in the 2 argument constructor\n");
    acc = *((const float4*)p);
    acc.x += c;
    acc.y += c;
    acc.z += c;
    acc.w += c;
  }

  DEVICE_INLINE Vec4T(const double* p) {
    acc.x = p[0];
    acc.y = p[1];
    acc.z = p[2];
    acc.w = p[3];
  }

  DEVICE_INLINE void store(float* p) {
    *((float4*)p) = acc;
  }
  DEVICE_INLINE void store(at::Half* p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }
  
  DEVICE_INLINE void element_wise_exp_(Vec4T<float> a, float beta) {
  acc.x = std::exp(a.acc.x + beta);
  acc.y = std::exp(a.acc.y + beta);
  acc.z = std::exp(a.acc.z + beta);
  acc.w = std::exp(a.acc.w + beta);
  } 

};

template <typename DTYPE, typename CAST_DTYPE>
HOST_AND_DEVICE void ComputeGradientsElement(
    int bTgt,
    int t,
    int u,
    int maxSrcLen,
    int maxTgtLen,
    int numTargets,
    int blank,
    CAST_DTYPE clamp,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    const CAST_DTYPE* denominators,
    const CAST_DTYPE* alphas,
    const CAST_DTYPE* betas,
    DTYPE* gradients,
    int H = 1,
    bool fusedLogSmax = true) {
  const int& maxT = maxSrcLen;
  const int& maxU = maxTgtLen;
  const int& D = numTargets;

  const int bSrc = bTgt / H;
  const int T = srcLengths[bSrc];
  const int U = tgtLengths[bTgt] + 1;
// printf("Hello computegradientselement %d \n", numTargets);
  if (t >= T || u >= U) { // out of boundary.
    if (gradients == logits && t < maxT && u < maxU) {
      // gradients and logits are pointing to the same memory location
      Indexer3D idxr3(maxT, maxU);
      int idx_b_t_u_zero = idxr3(bTgt, t, u);
      if (idx_b_t_u_zero != -1) {
        int start = idx_b_t_u_zero * D;
        for (int b_t_u_d = start; b_t_u_d < start + D; ++b_t_u_d) {
          gradients[b_t_u_d] = 0;
        }
      }
    }
    return;
  }

  int costIdx = bTgt * maxT * maxU;
  CAST_DTYPE cost = -(betas[costIdx]);

  Indexer2D idxr2(maxU - 1);

  int idx_b_t_u, idx_b_t_up1, idx_b_tp1_u;
  Indexer3D idxr3(maxT, maxU);
  idx_b_t_u = idxr3(bTgt, t, u);
  idx_b_t_up1 = idxr3(bTgt, t, u + 1);
  idx_b_tp1_u = idxr3(bTgt, t + 1, u);

  if (idx_b_t_u == -1) {
    return;
  }

  if (isinf(cost) || isnan(cost)) {
    for (int d = 0; d < D; ++d) {
      int b_t_u_d = idx_b_t_u * D + d;
      gradients[b_t_u_d] = 0;
    }
    return;
  }

  CAST_DTYPE c = alphas[idx_b_t_u] + cost - denominators[idx_b_t_u];
  DTYPE grad;
  if (fusedLogSmax) {
  for (int d = 0; d*4 < D; d++) {
  //for (int d = 0; d < D; d++) {
    //int b_t_u_d = idx_b_t_u * D + d;
     int b_t_u_d = idx_b_t_u * D + d*4;
    //CAST_DTYPE g = CAST_DTYPE(logits[b_t_u_d]) + c;
    Vec4T<CAST_DTYPE> g_vec(reinterpret_cast<const CAST_DTYPE*>(&logits[b_t_u_d]), CAST_DTYPE(c));
    Vec4T<CAST_DTYPE> grad_vec;
    grad_vec.element_wise_exp_(g_vec, betas[idx_b_t_u]);
    grad_vec.store(&gradients[b_t_u_d]);
    
    //grad = std::exp(g + betas[idx_b_t_u]);
      // if (d == blank && t == T - 1 && u == U - 1) { // last blank transition.
      // //printf("a1\n");
      //   grad = grad - std::exp(g);
      // } else if (t < T - 1 && d == blank && idx_b_tp1_u != -1) {
      //   //printf("b1\n");
      //     grad = grad - std::exp(g + betas[idx_b_tp1_u]);
      // } else if (u < U - 1 && d == targets[idxr2(bTgt, u)] && idx_b_t_up1 != -1) {
      //  // printf("c1\n");
      //     grad = grad - std::exp(g + betas[idx_b_t_up1]);
      // }
    
    if (clamp > 0) {
      printf("clamp\n");
      auto g = CAST_DTYPE(grad);
      grad = math::min(g, clamp);
      grad = math::max(g, -clamp);
    } 
    //gradients[b_t_u_d] = grad;
  } 
  } else {
  for (int d = 0; d < D; ++d) {
    int b_t_u_d = idx_b_t_u * D + d;

   // Non fused log softmax case
      CAST_DTYPE g = cost + CAST_DTYPE(logits[b_t_u_d]);
      if (d == blank && t == T - 1 && u == U - 1) {
        gradients[b_t_u_d] = g + alphas[idx_b_t_u];
      } else if (t < T - 1 && d == blank) {
        if (idx_b_tp1_u != -1) {
          gradients[b_t_u_d] = g + alphas[idx_b_t_u] + betas[idx_b_tp1_u];
        } else {
          gradients[b_t_u_d] = g + CAST_DTYPE(-INFINITY);
        }
      } else if (u < U - 1 && d == targets[idxr2(bTgt, u)]) {
        if (idx_b_t_up1 != -1) {
          gradients[b_t_u_d] = g + alphas[idx_b_t_u] + betas[idx_b_t_up1];
        } else {
          gradients[b_t_u_d] = g + CAST_DTYPE(-INFINITY);
        }
      } else {
        gradients[b_t_u_d] = g + CAST_DTYPE(-INFINITY);
      }
      gradients[b_t_u_d] = -std::exp(gradients[b_t_u_d]);

    if (clamp > 0) {
      auto g = CAST_DTYPE(gradients[b_t_u_d]);
      gradients[b_t_u_d] = math::min(g, clamp);
      gradients[b_t_u_d] = math::max(g, -clamp);
    }
  }
  }
}

} // namespace rnnt
} // namespace torchaudio
