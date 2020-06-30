/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/dataset.h"

#include <unordered_map>
#include <fstream>
#include <iostream>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {
namespace {

// A wrapper class for storing a `DatasetBase` instance in a DT_VARIANT tensor.
// Objects of the wrapper class own a reference on an instance of `DatasetBase`,
// and the wrapper's copy constructor and destructor take care of managing the
// reference count.
//
// NOTE(mrry): This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot currently serialize an arbitrary
// `DatasetBase` object, so the `Encode()` and `Decode()` methods are not
// implemented.
class DatasetVariantWrapper {
 public:
  DatasetVariantWrapper() : dataset_(nullptr) {}

  // Transfers ownership of `dataset` to `*this`.
  explicit DatasetVariantWrapper(DatasetBase* dataset) : dataset_(dataset) {}

  DatasetVariantWrapper(const DatasetVariantWrapper& other)
      : dataset_(other.dataset_) {
    if (dataset_) dataset_->Ref();
  }

  DatasetVariantWrapper& operator=(DatasetVariantWrapper&& other) {
    if (&other == this) return *this;
    std::swap(dataset_, other.dataset_);
    return *this;
  }

  DatasetVariantWrapper& operator=(const DatasetVariantWrapper& other) = delete;

  ~DatasetVariantWrapper() {
    if (dataset_) dataset_->Unref();
  }

  DatasetBase* get() const { return dataset_; }

  string TypeName() const { return "tensorflow::DatasetVariantWrapper"; }
  string DebugString() const {
    if (dataset_) {
      return dataset_->DebugString();
    } else {
      return "<Uninitialized DatasetVariantWrapper>";
    }
  }
  void Encode(VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
  }
  bool Decode(const VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
    return false;
  }

 private:
  DatasetBase* dataset_;  // Owns one reference.
};

const char kWrappedDatasetVariantTypeName[] =
    "tensorflow::data::WrappedDatasetVariant";

class WrappedDatasetVariantWrapper {
 public:
  WrappedDatasetVariantWrapper() {}

  explicit WrappedDatasetVariantWrapper(const Tensor& ds_tensor)
      : ds_tensor_(ds_tensor) {}

  Tensor get() const { return ds_tensor_; }

  string TypeName() const { return "tensorflow::WrappedDatasetVariantWrapper"; }

  string DebugString() const {
    return "tensorflow::WrappedDatasetVariantWrapper::DebugString";
  }

  void Encode(VariantTensorData* data) const {
    *(data->add_tensors()) = ds_tensor_;
  }

  bool Decode(const VariantTensorData& data) {
    ds_tensor_ = data.tensors(0);
    return true;
  }

 private:
  Tensor ds_tensor_;
};

class WrapDatasetVariantOp : public OpKernel {
 public:
  explicit WrapDatasetVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                tensor.dtype() == DT_VARIANT &&
                    TensorShapeUtils::IsScalar(tensor.shape()),
                errors::InvalidArgument(
                    "Dataset tensor must be a scalar of dtype DT_VARIANT."));
    DatasetBase* unused;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(tensor, &unused));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->scalar<Variant>()() = WrappedDatasetVariantWrapper(tensor);
  }
};

REGISTER_KERNEL_BUILDER(Name("WrapDatasetVariant").Device(DEVICE_CPU),
                        WrapDatasetVariantOp);
REGISTER_KERNEL_BUILDER(Name("WrapDatasetVariant")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle")
                            .Device(DEVICE_GPU),
                        WrapDatasetVariantOp);

class UnwrapDatasetVariantOp : public OpKernel {
 public:
  explicit UnwrapDatasetVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                tensor.dtype() == DT_VARIANT &&
                    TensorShapeUtils::IsScalar(tensor.shape()),
                errors::InvalidArgument(
                    "Dataset tensor must be a scalar of dtype DT_VARIANT."));
    Variant variant = tensor.scalar<Variant>()();
    const WrappedDatasetVariantWrapper* wrapper =
        variant.get<WrappedDatasetVariantWrapper>();
    OP_REQUIRES(ctx, wrapper != nullptr,
                errors::InvalidArgument(
                    "Tensor must be a WrappedDataset variant object."));
    Tensor ds_tensor = wrapper->get();
    OP_REQUIRES_OK(ctx, ctx->set_output("output_handle", ds_tensor));
  }
};

REGISTER_KERNEL_BUILDER(Name("UnwrapDatasetVariant").Device(DEVICE_CPU),
                        UnwrapDatasetVariantOp);
REGISTER_KERNEL_BUILDER(Name("UnwrapDatasetVariant")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle")
                            .Device(DEVICE_GPU),
                        UnwrapDatasetVariantOp);

static Status WrappedDatasetVariantDeviceCopy(
    const WrappedDatasetVariantWrapper& from, WrappedDatasetVariantWrapper* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  *to = WrappedDatasetVariantWrapper(from);
  return Status::OK();
}

#define REGISTER_OPTIONAL_COPY(DIRECTION)               \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      WrappedDatasetVariantWrapper, DIRECTION,          \
      WrappedDatasetVariantDeviceCopy)

REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(WrappedDatasetVariantWrapper,
                                       kWrappedDatasetVariantTypeName);

}  // namespace

Status GraphDefBuilderWrapper::AddDataset(
    const DatasetBase* dataset,
    const std::vector<std::pair<size_t, Node*>>& inputs,
    const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    Node** output) {
  const string& type_string = dataset->type_string();
  std::unique_ptr<const GraphDefBuilder::Options> opts(
      new GraphDefBuilder::Options(b_->opts()));
  // TODO(srbs|mrry): Not all datasets have output_types and output_shapes
  // attributes defined. It will be nice to have a consistent pattern.
  bool has_output_types_attr = HasAttr(type_string, "output_types");
  bool has_output_shapes_attr = HasAttr(type_string, "output_shapes");
  if (has_output_shapes_attr) {
    opts.reset(new GraphDefBuilder::Options(
        opts->WithAttr("output_shapes", dataset->output_shapes())));
  }
  if (has_output_types_attr) {
    opts.reset(new GraphDefBuilder::Options(
        opts->WithAttr("output_types", dataset->output_dtypes())));
  }
  for (auto attr : attrs) {
    opts.reset(
        new GraphDefBuilder::Options(opts->WithAttr(attr.first, attr.second)));
  }
  if (opts->HaveError()) {
    return errors::Internal("AddDataset: Failed to build Options with error ",
                            opts->StatusToString());
  }
  NodeBuilder node_builder(opts->GetNameForOp(type_string), type_string,
                           opts->op_registry());
  {
    size_t total_size = inputs.size() + list_inputs.size();
    auto inputs_iter = inputs.begin();
    auto list_inputs_iter = list_inputs.begin();
    for (int i = 0; i < total_size; i++) {
      if (inputs_iter != inputs.end() && inputs_iter->first == i) {
        node_builder.Input(NodeBuilder::NodeOut(inputs_iter->second));
        inputs_iter++;
      } else if (list_inputs_iter != list_inputs.end() &&
                 list_inputs_iter->first == i) {
        std::vector<NodeBuilder::NodeOut> nodeout_inputs;
        nodeout_inputs.reserve(list_inputs_iter->second.size());
        for (Node* n : list_inputs_iter->second) {
          nodeout_inputs.emplace_back(n);
        }
        node_builder.Input(nodeout_inputs);
        list_inputs_iter++;
      } else {
        return errors::InvalidArgument("No input found for index ", i);
      }
    }
  }
  *output = opts->FinalizeBuilder(&node_builder);
  if (*output == nullptr) {
    return errors::Internal("AddDataset: Failed to build ", type_string,
                            " op with error ", opts->StatusToString());
  }
  return Status::OK();
}

Status GraphDefBuilderWrapper::AddFunction(
    SerializationContext* ctx, const string& function_name,
    const FunctionLibraryDefinition& lib_def) {
  if (b_->HasFunction(function_name)) {
    VLOG(1) << "Function with name " << function_name << "already exists in"
            << " the graph. It will not be added again.";
    return Status::OK();
  }
  if (!ctx->optimization_only()) {
    TF_RETURN_IF_ERROR(EnsureFunctionIsStateless(function_name, lib_def));
  }
  const FunctionDef* f_def = lib_def.Find(function_name);
  if (f_def == nullptr) {
    return errors::InvalidArgument("Unable to find FunctionDef for ",
                                   function_name, " in the registry.");
  }
  FunctionDefLibrary def;
  *def.add_function() = *f_def;
  const string gradient_func = lib_def.FindGradient(function_name);
  if (!gradient_func.empty()) {
    GradientDef* g_def = def.add_gradient();
    g_def->set_function_name(function_name);
    g_def->set_gradient_func(gradient_func);
  }
  TF_RETURN_IF_ERROR(b_->AddFunctionLibrary(def));

  // Recursively add functions in inputs of function_name.
  for (const NodeDef& node_def : f_def->node_def()) {
    const OpRegistrationData* op_reg_data = nullptr;
    TF_RETURN_IF_ERROR(lib_def.LookUp(node_def.op(), &op_reg_data));
    if (op_reg_data->is_function_op) {
      TF_RETURN_IF_ERROR(AddFunction(ctx, op_reg_data->op_def.name(), lib_def));
    }
    // Recursively add functions in attrs of this NodeDef.
    for (const auto& pair : node_def.attr()) {
      TF_RETURN_IF_ERROR(AddAttrFunctions(ctx, pair.second, lib_def));
    }
  }

  // Recursively add functions in attrs of function_name.
  for (auto iter = f_def->attr().begin(); iter != f_def->attr().end(); iter++) {
    TF_RETURN_IF_ERROR(AddAttrFunctions(ctx, iter->second, lib_def));
  }
  return Status::OK();
}

void GraphDefBuilderWrapper::AddPlaceholderInternal(const Tensor& val,
                                                    Node** output) {
  *output = ops::SourceOp(
      "Placeholder",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("shape", val.shape()));
}

void GraphDefBuilderWrapper::AddTensorInternal(const Tensor& val,
                                               Node** output) {
  *output = ops::SourceOp(
      "Const",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("value", val));
}

bool GraphDefBuilderWrapper::HasAttr(const string& name,
                                     const string& attr_name) const {
  const OpDef* op_def = nullptr;
  Status s = b_->opts().op_registry()->LookUpOpDef(name, &op_def);
  if (!s.ok() || op_def == nullptr) {
    return false;
  }
  return HasAttr(op_def, attr_name);
}

int64 GetAllocatedBytes(const std::vector<Tensor>& element) {
  int64 allocated_bytes = 0;
  DatasetBase* dataset;
  for (auto& tensor : element) {
    if (tensor.dtype() == DT_VARIANT &&
        GetDatasetFromVariantTensor(tensor, &dataset).ok()) {
      allocated_bytes += dataset->AllocatedBytes();
    } else {
      allocated_bytes += tensor.AllocatedBytes();
    }
  }
  return allocated_bytes;
}

Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset) {
  if (!(tensor.dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  const Variant& variant = tensor.scalar<Variant>()();
  const DatasetVariantWrapper* wrapper = variant.get<DatasetVariantWrapper>();
  if (wrapper == nullptr) {
    return errors::InvalidArgument("Tensor must be a Dataset object.");
  }
  *out_dataset = wrapper->get();
  if (*out_dataset == nullptr) {
    return errors::Internal("Read uninitialized Dataset variant.");
  }
  return Status::OK();
}

Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor) {
  if (!(tensor->dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor->shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<Variant>()() = DatasetVariantWrapper(dataset);
  return Status::OK();
}

Status DatasetBase::Save(SerializationContext* ctx,
                         IteratorStateWriter* writer) const {
  string serialized_graph_def;
  string output_node;
  GraphDefBuilder b;
  DatasetGraphDefBuilder db(&b);
  Node* node = nullptr;
  TF_RETURN_IF_ERROR(AsGraphDefInternal(ctx, &db, &node));
  output_node = node->name();
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(b.ToGraphDef(&graph_def));
  graph_def.SerializeToString(&serialized_graph_def);
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(kDatasetGraphKey, serialized_graph_def));
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(kDatasetGraphOutputNodeKey, output_node));
  return Status::OK();
}

Status DatasetBase::DatasetGraphDefBuilder::AddInputDataset(
    SerializationContext* ctx, const DatasetBase* dataset, Node** output) {
  Status status = dataset->AsGraphDefInternal(ctx, this, output);
  if (ctx->optimization_only() && errors::IsUnimplemented(status)) {
    Tensor t(DT_VARIANT, TensorShape({}));
    // `StoreDatasetInVariantTensor` will transfer ownership of `dataset`. We
    // increment the refcount of `dataset` here to retain ownership.
    dataset->Ref();
    TF_RETURN_IF_ERROR(
        StoreDatasetInVariantTensor(const_cast<DatasetBase*>(dataset), &t));
    TF_RETURN_IF_ERROR(AddPlaceholder(t, output));
    DCHECK_NE(ctx->input_list(), nullptr);
    ctx->input_list()->emplace_back((*output)->name(), std::move(t));
    LOG(WARNING)
        << "Input of " << dataset->DebugString()
        << " will not be optimized because the dataset does not implement the "
           "AsGraphDefInternal() method needed to apply optimizations.";
    return Status::OK();
  }
  return status;
}

void IndexManager::RecordFinished(EparallaxTensorIndex* index) {
  //uint64 start = Env::Default()->NowMicros();
  mutex_lock l(*mu_);
  bool all_processed;
  std::vector<EparallaxTensorIndex*> processed_indices_buffer;
  processed_indices_buffer.push_back(index);

  while (!processed_indices_buffer.empty()) {
    all_processed = true;
    EparallaxTensorIndex* processed_index = processed_indices_buffer.back();
    processed_indices_buffer.pop_back();

    {
      processed_indices_->Push(processed_index);
      RemoveChildren(processed_index);

      if (!IsOneToManyOp(processed_index->iterator_id())) {
        for (auto parent_index : *processed_index->parent_indices()) {
          processed_indices_buffer.push_back(parent_index);
        }
      } else {
        for (auto parent_index : *processed_index->parent_indices()) {
          if (infertile_indices_->Contains(parent_index) &&
              ChildrenAllProcessed(parent_index) &&
              !processed_indices_->Contains(parent_index)) {
            processed_indices_buffer.push_back(parent_index);
          }
        }
      }
    }
  }
  //LOG(INFO) << "RecordFinished took " << Env::Default()->NowMicros() - start << " usecs.";
}

void IndexManager::RecordInfertile(EparallaxTensorIndex* index) {
  mutex_lock l(*mu_);
  infertile_indices_->Push(index);
}

EparallaxTensorIndex* IndexManager::IssueNewIndex(
    string prefix, std::vector<EparallaxTensorIndex*>* parent_indices) {
  //uint64 start = Env::Default()->NowMicros();
  mutex_lock l(*mu_);
  EparallaxTensorIndex* out_index;
  {

    int64 last_local_index = -1;
    for (auto issued_index : *issued_indices_->Get(
          prefix, ToString(*parent_indices))) {
      if (*issued_index->parent_indices() == *parent_indices &&
          issued_index->local_index() > last_local_index) {
        last_local_index = issued_index->local_index();
      }
    }
    out_index = new EparallaxTensorIndex(prefix, parent_indices,
                                         last_local_index + 1);

    issued_indices_->Push(out_index);
  }
  //LOG(INFO) << prefix << " IssueNewIndex1 took " << Env::Default()->NowMicros() - start << " usecs.";
  //start = Env::Default()->NowMicros();
  for (auto parent_index : *out_index->parent_indices()) {
    auto it = children_indices_->find(parent_index->ToString());
    std::vector<EparallaxTensorIndex*>* q;
    if (it == children_indices_->end()) {
      q = new std::vector<EparallaxTensorIndex*>;
      children_indices_->insert(std::make_pair(parent_index->ToString(), q));
    } else {
      q = it->second;
    }
    q->push_back(out_index);
  }

  //LOG(INFO) << prefix << " IssueNewIndex2 took " << Env::Default()->NowMicros() - start << " usecs.";
  return out_index;
}

bool IndexManager::AlreadyProcessed(EparallaxTensorIndex* index) {
  mutex_lock l(*mu_);
  return processed_indices_->Contains(index);
}

void IndexManager::ResetIndex(string iterator_id) {
  mutex_lock l(*mu_);
  {
    processed_indices_->Clear(iterator_id);
  }
  {
    issued_indices_->Clear(iterator_id);
  }
  {
    infertile_indices_->Clear(iterator_id);
  }

  std::ofstream ckpt_file;
  string ckpt_file_path = string(ckpt_dir_) + "/index_ckpt_" +
      std::to_string(shard_index_);
  ckpt_file.open(ckpt_file_path.data());
  if (ckpt_file.is_open()) {
    ckpt_file << "\n";
    ckpt_file.close();
  }
}

void IndexManager::SetShardID(int64 index) {
  shard_index_ = index;
}

bool IndexManager::IsFirstCall(string iterator_id) {
  mutex_lock l(*mu_);
  return issued_indices_->Empty(iterator_id) && processed_indices_->Empty();
}

Status DatasetBaseIterator::GetNextFromInput(
    IteratorBase* const input_impl, IteratorContext* ctx,
    std::vector<Tensor>* out_tensors,
    bool* end_of_sequence,
    std::vector<EparallaxTensorIndex*>* parent_indices) {
  EparallaxTensorIndex* out_index;
  Status s = input_impl->GetNext(ctx, out_tensors, end_of_sequence, out_index);
  if (s.ok() && !*end_of_sequence && parent_indices != nullptr) {
    parent_indices->push_back(out_index);
  }
  return s;
}

Status DatasetBaseIterator::GetNext(IteratorContext* ctx,
                                    std::vector<Tensor>* out_tensors,
                                    bool* end_of_sequence,
                                    EparallaxTensorIndex*& out_index) {
  profiler::TraceMe activity([&] { return BuildTraceMeName(); },
                             profiler::TraceMeLevel::kInfo);
  RecordStart(ctx, /*stop_output=*/true);
  std::vector<EparallaxTensorIndex*>* parent_indices =
      new std::vector<EparallaxTensorIndex*>;
  Status s = GetNextInternal(ctx, out_tensors, end_of_sequence, parent_indices);
  out_index = ctx->index_manager()->IssueNewIndex(prefix(), parent_indices);

  // `out_tensors` is empty if the parent indices have been already processed.
  if (!s.ok() || *end_of_sequence || out_tensors->empty()) {
    return s;
  }
  if (ctx->index_manager()->AlreadyProcessed(out_index)) {
    out_tensors->clear();
    return s;
  }

  if (s.ok() && !*end_of_sequence) RecordElement(ctx);
  RecordStop(ctx, /*start_output=*/true);
  if (TF_PREDICT_FALSE(errors::IsOutOfRange(s))) {
    s = errors::Internal("Iterator \"", params_.prefix,
                         "\" returned `OutOfRange`. This indicates an "
                         "implementation error as `OutOfRange` errors are not "
                         "expected to be returned here. Original message: ",
                         s.error_message());
    LOG(ERROR) << s;
  }
  return s;
}

void DatasetOpKernel::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset = nullptr;
  MakeDataset(ctx, &dataset);
  if (ctx->status().ok()) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, StoreDatasetInVariantTensor(dataset, output));
  }
}

void UnaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  MakeDataset(ctx, input, output);
}

void BinaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  DatasetBase* another_input;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(1), &another_input));
  MakeDataset(ctx, input, another_input, output);
}

const char DatasetBase::kDatasetGraphKey[] = "_DATASET_GRAPH";
const char DatasetBase::kDatasetGraphOutputNodeKey[] =
    "_DATASET_GRAPH_OUTPUT_NODE";

BackgroundWorker::BackgroundWorker(Env* env, const string& name) {
  thread_.reset(env->StartThread({} /* thread_options */, name,
                                 [this]() { WorkerLoop(); }));
}

BackgroundWorker::~BackgroundWorker() {
  {
    mutex_lock l(mu_);
    cancelled_ = true;
  }
  cond_var_.notify_one();
  // Block until the background thread has terminated.
  //
  // NOTE(mrry): We explicitly free and join the thread here because
  // `WorkerLoop()` uses other members of this object, and so we must join
  // the thread before destroying them.
  thread_.reset();
}

void BackgroundWorker::Schedule(std::function<void()> work_item) {
  {
    mutex_lock l(mu_);
    work_queue_.push_back(std::move(work_item));
  }
  cond_var_.notify_one();
}

void BackgroundWorker::WorkerLoop() {
  while (true) {
    std::function<void()> work_item = nullptr;
    {
      mutex_lock l(mu_);
      while (!cancelled_ && work_queue_.empty()) {
        cond_var_.wait(l);
      }
      if (cancelled_) {
        return;
      }
      DCHECK(!work_queue_.empty());
      work_item = std::move(work_queue_.front());
      work_queue_.pop_front();
    }
    DCHECK(work_item != nullptr);
    work_item();
  }
}

namespace {
class RunnerImpl : public Runner {
 public:
  void Run(const std::function<void()>& f) override {
    f();

    // NOTE: We invoke a virtual function to prevent `f` being tail-called, and
    // thus ensure that this function remains on the stack until after `f`
    // returns.
    PreventTailCall();
  }

 private:
  virtual void PreventTailCall() {}
};
}  // namespace

/* static */
Runner* Runner::get() {
  static Runner* singleton = new RunnerImpl;
  return singleton;
}

}  // namespace data
}  // namespace tensorflow
