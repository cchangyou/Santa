#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GReLUForward(const int n, const Dtype* in, Dtype* out,
    const Dtype sigma, const Dtype* pdfdivcdf_table, const Dtype* pdfdivcdf_table_g37) {
  CUDA_KERNEL_LOOP(index, n) {
    //Dtype ratio;
    Dtype tmp = in[index] / sigma;
//    if(tmp <= Dtype(0)){//-100
    if(tmp <= Dtype(-4999)){//for fnn
      if(tmp < -100){
        out[index] = Dtype(0);
      }else{
        Dtype f0 = Dtype(0.797884560802865 * sigma / 100);
        out[index] = (100 + tmp) * f0;
      }
    }else if(tmp > Dtype(37)){//>37
      out[index] = in[index];
    }else{
      Dtype ratio;
      if(tmp < -37){
        int zeta0 = (int)tmp;//(int)tmp - 1;
	int idx = 37 - zeta0;
	if(idx < 0) idx = 0;
	if(idx > 4963) idx = 4963;
        ratio = -tmp + Dtype(zeta0 + pdfdivcdf_table[idx]);
      }else{
	int idx = (int)(tmp * 10000) + 370000;
	if(idx < 0) idx = 0;
	if(idx > 740000) idx = 740000;
        ratio = pdfdivcdf_table_g37[idx];
      }
      out[index] = in[index] + sigma * ratio;
    }

/*    if(zeta < Dtype(-37)){
      if(zeta < Dtype(-4999)){
        ratio = -zeta;
      }else{
        int zeta0 = (int)zeta;
        ratio = -zeta + Dtype(zeta0 + pdfdivcdf_table[37 - zeta0]);
      }
    }else if(zeta >= Dtype(-37) && zeta <= Dtype(37)){
      ratio = pdfdivcdf_table_g37[(int)(zeta * 10000) + 370000];
    }else{
      ratio = Dtype(0);
    }
    out[index] = in[index] + sigma * ratio;*/
  }
}

template <typename Dtype>
void GReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype* pdfdivcdf_table = tables[0]->mutable_gpu_data();
  const Dtype* pdfdivcdf_table_g37 = tables[1]->mutable_gpu_data();
  //const Dtype* ss = tables[2]->mutable_gpu_data();
  const Dtype sigma = this->layer_param_.grelu_param().sigma();//Dtype(0.0001);//tables[2]->mutable_gpu_data()[0];

  // NOLINT_NEXT_LINE(whitespace/operators)
  GReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, sigma, pdfdivcdf_table, pdfdivcdf_table_g37);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void GReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype sigma, const Dtype* pdfdivcdf_table, const Dtype* pdfdivcdf_table_g37) {
  CUDA_KERNEL_LOOP(index, n) {

    Dtype tmp = in_data[index] / sigma;
//    if(tmp <= Dtype(0)){//-100
    if(tmp <= Dtype(-4999)){
      //out_diff[index] = Dtype(0);
      if(tmp < -100){ 
        out_diff[index] = Dtype(0);
      }else{
        out_diff[index] = Dtype(0.797884560802865 * sigma / 100) * in_diff[index];
      }
    }else if(tmp > Dtype(37)){
      out_diff[index] = in_diff[index];
    }else{
      Dtype ratio;
      if(tmp < -37){
        int zeta0 = (int)tmp;//(int)tmp - 1;
	int idx = -37 - zeta0;
	if(idx < 0) idx = 0;
	if(idx > 4963) idx = 4963;
        ratio = -tmp + Dtype(zeta0 + pdfdivcdf_table[idx]);
      }else{
	int idx = (int)(tmp * 10000) + 370000;
	if(idx < 0) idx = 0;
	if(idx > 740000) idx = 740000;
        ratio = pdfdivcdf_table_g37[idx];
      }
      out_diff[index] = in_diff[index] * (Dtype(1) - tmp * ratio - ratio * ratio);
    }

/*    Dtype ratio;
    Dtype zeta = in_data[index] / sigma;
    if(zeta < Dtype(-37)){
      if(zeta < Dtype(-4999)){
        ratio = -zeta;
      }else{
        int zeta0 = (int)zeta;
        ratio = -zeta + Dtype(zeta0 + pdfdivcdf_table[37 - zeta0]);
      }
    }else if(zeta >= Dtype(-37) && zeta <= Dtype(37)){
      ratio = pdfdivcdf_table_g37[(int)(zeta * 10000) + 370000];
    }else{
      ratio = Dtype(0);
    }
    out_diff[index] = in_diff[index] * (Dtype(1) - in_data[index] / sigma * ratio
        - ratio * ratio);*/
  }
}

template <typename Dtype>
void GReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype* pdfdivcdf_table = tables[0]->mutable_gpu_data();
    const Dtype* pdfdivcdf_table_g37 = tables[1]->mutable_gpu_data();
    //const Dtype* ss = tables[2]->mutable_gpu_data();
    const Dtype sigma = this->layer_param_.grelu_param().sigma();//Dtype(0.0001);//tables[2]->mutable_gpu_data()[0];

    // NOLINT_NEXT_LINE(whitespace/operators)
    GReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, sigma, pdfdivcdf_table, pdfdivcdf_table_g37);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(GReLULayer);


}  // namespace caffe
