#pragma once

// @generated by torchgen/gen.py from NativeMetaFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <tuple>
#include <vector>

namespace at {
namespace meta {

struct TORCH_API structured_cat : public at::impl::MetaBase {
    
                template <bool DIM = false, bool VALID = false, bool ALL_CONTIGUOUS = false, bool ALL_SAME_DTYPE = false, bool ALL_SAME_SIZES_AND_STRIDE = false, bool MEMORY_FORMAT = false>
                struct TORCH_API precompute_out {
                    
                    precompute_out<true, VALID, ALL_CONTIGUOUS, ALL_SAME_DTYPE, ALL_SAME_SIZES_AND_STRIDE, MEMORY_FORMAT> set_dim(int64_t value) {
                        static_assert(DIM == false, "dim already set");
                        precompute_out<true, VALID, ALL_CONTIGUOUS, ALL_SAME_DTYPE, ALL_SAME_SIZES_AND_STRIDE, MEMORY_FORMAT> ret;
ret.dim = value;
ret.valid = this->valid;
ret.all_contiguous = this->all_contiguous;
ret.all_same_dtype = this->all_same_dtype;
ret.all_same_sizes_and_stride = this->all_same_sizes_and_stride;
ret.memory_format = this->memory_format;
return ret;
                    }
                

                    precompute_out<DIM, true, ALL_CONTIGUOUS, ALL_SAME_DTYPE, ALL_SAME_SIZES_AND_STRIDE, MEMORY_FORMAT> set_valid(int64_t value) {
                        static_assert(VALID == false, "valid already set");
                        precompute_out<DIM, true, ALL_CONTIGUOUS, ALL_SAME_DTYPE, ALL_SAME_SIZES_AND_STRIDE, MEMORY_FORMAT> ret;
ret.dim = this->dim;
ret.valid = value;
ret.all_contiguous = this->all_contiguous;
ret.all_same_dtype = this->all_same_dtype;
ret.all_same_sizes_and_stride = this->all_same_sizes_and_stride;
ret.memory_format = this->memory_format;
return ret;
                    }
                

                    precompute_out<DIM, VALID, true, ALL_SAME_DTYPE, ALL_SAME_SIZES_AND_STRIDE, MEMORY_FORMAT> set_all_contiguous(bool value) {
                        static_assert(ALL_CONTIGUOUS == false, "all_contiguous already set");
                        precompute_out<DIM, VALID, true, ALL_SAME_DTYPE, ALL_SAME_SIZES_AND_STRIDE, MEMORY_FORMAT> ret;
ret.dim = this->dim;
ret.valid = this->valid;
ret.all_contiguous = value;
ret.all_same_dtype = this->all_same_dtype;
ret.all_same_sizes_and_stride = this->all_same_sizes_and_stride;
ret.memory_format = this->memory_format;
return ret;
                    }
                

                    precompute_out<DIM, VALID, ALL_CONTIGUOUS, true, ALL_SAME_SIZES_AND_STRIDE, MEMORY_FORMAT> set_all_same_dtype(bool value) {
                        static_assert(ALL_SAME_DTYPE == false, "all_same_dtype already set");
                        precompute_out<DIM, VALID, ALL_CONTIGUOUS, true, ALL_SAME_SIZES_AND_STRIDE, MEMORY_FORMAT> ret;
ret.dim = this->dim;
ret.valid = this->valid;
ret.all_contiguous = this->all_contiguous;
ret.all_same_dtype = value;
ret.all_same_sizes_and_stride = this->all_same_sizes_and_stride;
ret.memory_format = this->memory_format;
return ret;
                    }
                

                    precompute_out<DIM, VALID, ALL_CONTIGUOUS, ALL_SAME_DTYPE, true, MEMORY_FORMAT> set_all_same_sizes_and_stride(bool value) {
                        static_assert(ALL_SAME_SIZES_AND_STRIDE == false, "all_same_sizes_and_stride already set");
                        precompute_out<DIM, VALID, ALL_CONTIGUOUS, ALL_SAME_DTYPE, true, MEMORY_FORMAT> ret;
ret.dim = this->dim;
ret.valid = this->valid;
ret.all_contiguous = this->all_contiguous;
ret.all_same_dtype = this->all_same_dtype;
ret.all_same_sizes_and_stride = value;
ret.memory_format = this->memory_format;
return ret;
                    }
                

                    precompute_out<DIM, VALID, ALL_CONTIGUOUS, ALL_SAME_DTYPE, ALL_SAME_SIZES_AND_STRIDE, true> set_memory_format(at::MemoryFormat value) {
                        static_assert(MEMORY_FORMAT == false, "memory_format already set");
                        precompute_out<DIM, VALID, ALL_CONTIGUOUS, ALL_SAME_DTYPE, ALL_SAME_SIZES_AND_STRIDE, true> ret;
ret.dim = this->dim;
ret.valid = this->valid;
ret.all_contiguous = this->all_contiguous;
ret.all_same_dtype = this->all_same_dtype;
ret.all_same_sizes_and_stride = this->all_same_sizes_and_stride;
ret.memory_format = value;
return ret;
                    }
                
                    int64_t dim;
int64_t valid;
bool all_contiguous;
bool all_same_dtype;
bool all_same_sizes_and_stride;
at::MemoryFormat memory_format;
            };
    using meta_return_ty = precompute_out <true, true, true, true, true, true>;
    meta_return_ty meta(const at::ITensorListRef & tensors, int64_t dim);
};

} // namespace native
} // namespace at