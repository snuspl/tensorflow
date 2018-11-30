/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Custom Slow Conv2D Op

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <string.h>
#include <map>
#include <vector>

#include "tensorflow/core/kernels/conv2d_slow_op.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/deep_conv2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                  \
  template <>                                                                \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,        \
      typename TTypes<T, 4, int>::Tensor out);                               \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;              \
  template <>                                                                \
  void PadInput<GPUDevice, T, int, 4>::operator()(                           \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,        \
      const std::array<int, 2>& padding_left,                                \
      const std::array<int, 2>& padding_right,                               \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format);     \
  extern template struct PadInput<GPUDevice, T, int, 4>;                     \
  template<>                                                                 \
  void NHWCToNCHW<GPUDevice, T, 4>::operator()(                              \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor in,             \
      typename TTypes<T, 4>::Tensor out);                                    \
  extern template struct NHWCToNCHW<GPUDevice, T, 4>;                        \
  template<>                                                                 \
  void NCHWToNHWC<GPUDevice, T, 4>::operator()(                              \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor in,             \
      typename TTypes<T, 4>::Tensor out);                                    \
  extern template struct NCHWToNHWC<GPUDevice, T, 4>;
 
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// A dummy type to group forward convolution autotune results together.
struct ConvAutoTuneGroup {
  static string name() { return "Conv"; }
};
typedef AutoTuneSingleton<ConvAutoTuneGroup, ConvParameters,
                          se::dnn::AlgorithmConfig>
    AutoTuneConv;

template <typename T>
void LaunchConv2DSlowOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& input_param, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    Tensor* output, TensorFormat data_format, int slow_iter) {
  using se::dnn::AlgorithmConfig;
  using se::dnn::AlgorithmDesc;
  using se::dnn::ProfileResult;
  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  if (!use_cudnn) {
    ctx->SetStatus(
        errors::Unimplemented("Conv2D for GPU is not currently supported "
                              "without cudnn"));
    return;
  }

  Tensor input = input_param;
  const int64 in_batch = GetTensorDim(input, data_format, 'N');
  int64 in_rows = GetTensorDim(input, data_format, 'H');
  int64 in_cols = GetTensorDim(input, data_format, 'W');
  const int64 in_depths = GetTensorDim(input, data_format, 'C');
  const int64 patch_rows = filter.dim_size(0);
  const int64 patch_cols = filter.dim_size(1);
  const int64 patch_depths = filter.dim_size(2);

  // If the filter in-depth (patch_depths) is 1 and smaller than the input
  // depth, it's a depthwise convolution. More generally, if the filter in-depth
  // divides but is smaller than the input depth, it is a grouped convolution.
  bool is_grouped_convolution = patch_depths != in_depths;
  if (patch_rows == 1 && patch_cols == 1 && !is_grouped_convolution &&
      row_dilation == 1 && col_dilation == 1 && row_stride == 1 &&
      col_stride == 1 && data_format == FORMAT_NHWC) {
    // 1x1 filter, so call cublas directly.
    const uint64 m = in_batch * in_rows * in_cols;
    const uint64 k = patch_depths;
    const uint64 n = filter.dim_size(3);

    auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                output->template flat<T>().size());

    auto no_transpose = se::blas::Transpose::kNoTranspose;
    bool blas_launch_status =
        stream
            ->ThenBlasGemm(no_transpose, no_transpose, n, m, k, 1.0f, b_ptr, n,
                           a_ptr, k, 0.0f, &c_ptr, n)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                      ", n=", n, ", k=", k));
    }
    return;
  } else if (patch_rows == in_rows && patch_cols == in_cols &&
             !is_grouped_convolution && row_dilation == 1 &&
             col_dilation == 1 && padding == VALID &&
             data_format == FORMAT_NHWC) {
    // The input data and filter have the same height/width, so call cublas
    // directly.
    const uint64 m = in_batch;
    const uint64 k = patch_rows * patch_cols * patch_depths;
    const uint64 n = filter.dim_size(3);

    auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                filter.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                output->template flat<T>().size());

    auto no_transpose = se::blas::Transpose::kNoTranspose;
    bool blas_launch_status =
        stream
            ->ThenBlasGemm(no_transpose, no_transpose, n, m, k, 1.0f, b_ptr, n,
                           a_ptr, k, 0.0f, &c_ptr, n)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                      ", n=", n, ", k=", k));
    }
    return;
  }

  int padding_rows = 0;
  int padding_cols = 0;
  const int64 out_batch = GetTensorDim(*output, data_format, 'N');
  const int64 out_rows = GetTensorDim(*output, data_format, 'H');
  const int64 out_cols = GetTensorDim(*output, data_format, 'W');
  const int64 out_depths = GetTensorDim(*output, data_format, 'C');
  if (padding == SAME) {
    // Total padding on rows and cols is
    // Pr = (R' - 1) * S + (Kr - 1) * Dr + 1 - R
    // Pc = (C' - 1) * S + (Kc - 1) * Dc + 1 - C
    // where (R', C') are output dimensions, (R, C) are input dimensions, S
    // is stride, (Dr, Dc) are dilations, (Kr, Kc) are filter dimensions.
    // We pad Pr/2 on the left and Pr - Pr/2 on the right, Pc/2 on the top
    // and Pc - Pc/2 on the bottom.  When Pr or Pc is odd, this means
    // we pad more on the right and bottom than on the top and left.
    padding_rows =
        std::max<int>(0, (out_rows - 1) * row_stride +
                             (patch_rows - 1) * row_dilation + 1 - in_rows);
    padding_cols =
        std::max<int>(0, (out_cols - 1) * col_stride +
                             (patch_cols - 1) * col_dilation + 1 - in_cols);
    const bool rows_odd = (padding_rows % 2 != 0);
    const bool cols_odd = (padding_cols % 2 != 0);
    if (rows_odd || cols_odd) {
      Tensor transformed_input;
      int64 new_in_rows = in_rows + rows_odd;
      int64 new_in_cols = in_cols + cols_odd;
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(DataTypeToEnum<T>::value,
                             ShapeFromFormat(data_format, in_batch, new_in_rows,
                                             new_in_cols, in_depths),
                             &transformed_input));

      functor::PadInput<GPUDevice, T, int, 4>()(
          ctx->eigen_device<GPUDevice>(), To32Bit(input_param.tensor<T, 4>()),
          {{0, 0}}, {{rows_odd, cols_odd}},
          To32Bit(transformed_input.tensor<T, 4>()), data_format);

      input = transformed_input;
      in_rows = new_in_rows;
      in_cols = new_in_cols;
    }
  }

  if (data_format == FORMAT_NHWC) {
    // Convert the input tensor from NHWC to NCHW.
    TensorShape nchw_shape =
        ShapeFromFormat(FORMAT_NCHW, in_batch, in_rows, in_cols, in_depths);
    if (in_depths > 1) {
      Tensor transformed_input;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             nchw_shape, &transformed_input));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(input).tensor<T, 4>(),
          transformed_input.tensor<T, 4>());
      input = transformed_input;
    } else {
      // If depth <= 1, then just reshape.
      CHECK(input.CopyFrom(input, nchw_shape));
    }
  }

  CHECK(padding_rows >= 0 && padding_cols >= 0)
      << "Negative row or col paddings: (" << padding_rows << ", "
      << padding_cols << ")";
  se::dnn::BatchDescriptor input_desc;
  input_desc.set_count(in_batch)
      .set_feature_map_count(in_depths)
      .set_height(in_rows)
      .set_width(in_cols)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  se::dnn::BatchDescriptor output_desc;
  output_desc.set_count(out_batch)
      .set_height(out_rows)
      .set_width(out_cols)
      .set_feature_map_count(out_depths)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  se::dnn::FilterDescriptor filter_desc;
  filter_desc.set_input_filter_height(patch_rows)
      .set_input_filter_width(patch_cols)
      .set_input_feature_map_count(patch_depths)
      .set_output_feature_map_count(filter.dim_size(3));
  se::dnn::ConvolutionDescriptor conv_desc;
  conv_desc.set_vertical_dilation_rate(row_dilation)
      .set_horizontal_dilation_rate(col_dilation)
      .set_vertical_filter_stride(row_stride)
      .set_horizontal_filter_stride(col_stride)
      .set_zero_padding_height(padding_rows / 2)
      .set_zero_padding_width(padding_cols / 2)
      .set_group_count(in_depths / patch_depths);

  Tensor transformed_filter;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                          DataTypeToEnum<T>::value,
                          TensorShape({filter.dim_size(3), filter.dim_size(2),
                                       filter.dim_size(0), filter.dim_size(1)}),
                          &transformed_filter));

  functor::TransformFilter<GPUDevice, T, int, 4>()(
      ctx->eigen_device<GPUDevice>(), To32Bit(filter.tensor<T, 4>()),
      To32Bit(transformed_filter.tensor<T, 4>()));

  Tensor transformed_output;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                              ShapeFromFormat(FORMAT_NCHW, out_batch, out_rows,
                                              out_cols, out_depths),
                              &transformed_output));

  auto input_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());
  auto filter_ptr =
      AsDeviceMemory(transformed_filter.template flat<T>().data(),
                     transformed_filter.template flat<T>().size());
  auto output_ptr =
      AsDeviceMemory(transformed_output.template flat<T>().data(),
                     transformed_output.template flat<T>().size());

  static int64 ConvolveScratchSize = GetCudnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  int device_id = stream->parent()->device_ordinal();
  DataType dtype = input.dtype();
  ConvParameters conv_parameters = {
      in_batch,          // batch
      in_depths,         // in_depths
      {{in_rows,         // in_rows
        in_cols}},       // in_cols
      FORMAT_NCHW,       // compute_data_format
      out_depths,        // out_depths
      {{patch_rows,      // filter_rows
        patch_cols,      // filter_cols
        patch_depths}},  // filter_depths
      {{row_dilation,    // dilation_rows
        col_dilation}},  // dilation_cols
      {{row_stride,      // stride_rows
        col_stride}},    // stride_cols
      {{padding_rows,    // padding_rows
        padding_cols}},  // padding_cols
      dtype,             // tensor datatype
      device_id,         // device_id
  };
  AlgorithmConfig algorithm_config;
  if (cudnn_use_autotune &&
      !AutoTuneConv::GetInstance()->Find(conv_parameters, &algorithm_config)) {
    std::vector<AlgorithmDesc> algorithms;
    CHECK(stream->parent()->GetConvolveAlgorithms(
        conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(stream->parent()),
        &algorithms));
    ProfileResult best_result;
    ProfileResult best_result_no_scratch;
    for (auto profile_algorithm : algorithms) {
      // TODO(zhengxq): profile each algorithm multiple times to better
      // accuracy.
      CudnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
      ProfileResult profile_result;
      bool cudnn_launch_status =
          stream
              ->ThenConvolveWithAlgorithm(
                  input_desc, input_ptr, filter_desc, filter_ptr, conv_desc,
                  output_desc, &output_ptr, &scratch_allocator,
                  AlgorithmConfig(profile_algorithm), &profile_result)
              .ok();
      if (cudnn_launch_status) {
        if (profile_result.is_valid()) {
          if (profile_result.elapsed_time_in_ms() <
              best_result.elapsed_time_in_ms()) {
            best_result = profile_result;
          }
          if (scratch_allocator.TotalByteSize() == 0 &&
              profile_result.elapsed_time_in_ms() <
                  best_result_no_scratch.elapsed_time_in_ms()) {
            best_result_no_scratch = profile_result;
          }
        }
      }
    }
    // TODO(yangzihao): refactor the profile result checking code into a common
    // utility function.
    OP_REQUIRES(ctx,
                best_result.is_valid() || best_result_no_scratch.is_valid(),
                errors::NotFound("No algorithm worked!"));
    if (best_result.is_valid()) {
      algorithm_config.set_algorithm(best_result.algorithm());
    }
    if (best_result_no_scratch.is_valid()) {
      algorithm_config.set_algorithm_no_scratch(
          best_result_no_scratch.algorithm());
    }
    AutoTuneConv::GetInstance()->Insert(conv_parameters, algorithm_config);
  }

  CudnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
  bool cudnn_launch_status =
      stream
          ->ThenConvolveWithAlgorithm(input_desc, input_ptr, filter_desc,
                                      filter_ptr, conv_desc, output_desc,
                                      &output_ptr, &scratch_allocator,
                                      algorithm_config, nullptr)
          .ok();
  for (int i = 0; i < slow_iter - 1; i++) {
      stream
          ->ThenConvolveWithAlgorithm(input_desc, input_ptr, filter_desc,
                                      filter_ptr, conv_desc, output_desc,
                                      &output_ptr, &scratch_allocator,
                                      algorithm_config, nullptr)
          .ok();
  }

  if (!cudnn_launch_status) {
    ctx->SetStatus(errors::Internal(
        "cuDNN launch failure : input shape(", input.shape().DebugString(),
        ") filter shape(", filter.shape().DebugString(), ")"));
  }

  // Convert the output tensor back from NHWC to NCHW.
  if (data_format == FORMAT_NHWC) {
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        ctx->eigen_device<GPUDevice>(),
        const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
        output->tensor<T, 4>());
  } else {
    *output = transformed_output;
  }
}

// HJ: Manual Slow Conv Op
template <typename Device, typename T>
class Conv2DSlowOp : public BinaryOp<T> {
 public:
  explicit Conv2DSlowOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES_OK(context, context->GetAttr("slow_iter", &slow_iter));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    const int64 stride_h = GetTensorDim(strides_, data_format_, 'H');
    const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));

    const int64 dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    const int64 dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    const int64 dilation_h = GetTensorDim(dilations_, data_format_, 'H');
    const int64 dilation_w = GetTensorDim(dilations_, data_format_, 'W');
    OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h > 0 && dilation_w > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]

    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(
          context,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth or be evenly divisible by filter's in_depth.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    const int64 patch_depth = filter.dim_size(2);
    OP_REQUIRES(context, in_depth % patch_depth == 0,
                errors::InvalidArgument(
                    "input depth must be evenly divisible by filter depth: ",
                    in_depth, " vs ", patch_depth));

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride and dilation from the second and third
    // dimensions only (we do not support striding or dilation on the batch or
    // depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context, GetWindowedOutputSizeV2(
                                input_rows, filter_rows, dilation_rows,
                                stride_rows, padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeV2(
                                input_cols, filter_cols, dilation_cols,
                                stride_cols, padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(1) << "ConvSlow2D: in_depth = " << in_depth
            << ", patch_depth = " << patch_depth
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", dilation_rows = " << dilation_rows
            << ", dilation_cols = " << dilation_cols
            << ", out_depth = " << out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    launcher_(context, use_cudnn_, cudnn_use_autotune_, input, filter,
              dilation_rows, dilation_cols, stride_rows, stride_cols, padding_,
              output, data_format_, slow_iter);
  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  bool use_cudnn_;
  int slow_iter;
  Padding padding_;
  TensorFormat data_format_;
  LaunchConv2DSlowOp<Device, T> launcher_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DSlowOp);
};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv2DSlow").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv2DSlowOp<CPUDevice, T>);

#if GOOGLE_CUDA

int64 GetCudnnWorkspaceLimit(const string& envvar_in_mb,
                             int64 default_value_in_bytes) {
  const char* workspace_limit_in_mb_str = getenv(envvar_in_mb.c_str());
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64 scratch_limit_in_mb = -1;
    if (strings::safe_strto64(workspace_limit_in_mb_str,
                              &scratch_limit_in_mb)) {
      return scratch_limit_in_mb * (1 << 20);
    } else {
      LOG(WARNING) << "Invalid value for env-var " << envvar_in_mb << ": "
                   << workspace_limit_in_mb_str;
    }
  }
  return default_value_in_bytes;
}

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("Conv2DSlow").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    Conv2DSlowOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("Conv2DSlow").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    Conv2DSlowOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("Conv2DSlow").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    Conv2DSlowOp<GPUDevice, double>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
