#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AveAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.ave_accuracy_param().top_k();
  npartition = this->layer_param_.ave_accuracy_param().npartition();

  has_ignore_label_ =
    this->layer_param_.ave_accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.ave_accuracy_param().ignore_label();
  }
  burnin = this->layer_param_.ave_accuracy_param().burnin();
  /*label_axis_ =
        bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  int dim = bottom[0]->count() / outer_num_;
  int num_labels = bottom[0]->shape(label_axis_);
  int len = (outer_num_ - 1) * dim + (num_labels - 1) * inner_num_ + inner_num_;*/
  for(int j = 0; j < npartition; j++){
    vector<Dtype> v;
    for(int i = 0; i < this->layer_param_.ave_accuracy_param().nbatch()*this->layer_param_.ave_accuracy_param().nclass(); i++){
      v.push_back(Dtype(0));
    }
    hist_bottom_data.push_back(v);
  }
  
  hist_idx = 0; niter = 0;
}

template <typename Dtype>
void AveAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.ave_accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void AveAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
	
  int iter = niter / this->layer_param_.ave_accuracy_param().npartition();
  if(iter < burnin){
    fill(hist_bottom_data[hist_idx].begin(), hist_bottom_data[hist_idx].end(), 0);
  }
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
    	  hist_bottom_data[hist_idx][i * dim + k * inner_num_ + j] = hist_bottom_data[hist_idx][i * dim + k * inner_num_ + j]
    	                                                                    + bottom_data[i * dim + k * inner_num_ + j];
        bottom_data_vector.push_back(std::make_pair(
            hist_bottom_data[hist_idx][i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
          break;
        }
      }
      ++count;
    }
  }
  hist_idx = hist_idx + 1; hist_idx = hist_idx % npartition;
  niter++;

  // LOG(INFO) << "Accuracy: " << accuracy;
	//LOG(INFO) << "hist: " << hist_bottom_data[0] << ", outer_num_ = " << outer_num_ << ", inner_num_ = " << inner_num_ << ", count = " << bottom[0]->count();
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AveAccuracyLayer);
REGISTER_LAYER_CLASS(AveAccuracy);

}  // namespace caffe
