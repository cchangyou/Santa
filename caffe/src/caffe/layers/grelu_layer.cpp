#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void GReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  NeuronLayer<Dtype>::LayerSetUp(bottom, top);

  Blob<Dtype>* blob1 = new Blob<Dtype>(4964, 1, 1, 1);
  Dtype* pdfdivcdf_table = new Dtype[4964];
  //sigma = Dtype(0.01);
  //sigma = this->layer_param_.grelu_param().sigma();
  std::ifstream ifile("pdfdivcdf_table.txt", std::ios::in);
  std::ifstream ifile1("pdfdivcdf_table_g37.txt", std::ios::in);
  Dtype output;
  int n = 0;
  if (ifile.is_open()) {
    while (ifile >> output) {
      pdfdivcdf_table[n] = output;
      n++;
     }
  }else{
    std::cerr << "There was a problem opening the input file pdfdivcdf_table.txt!\n";
    exit(1);
  }
  blob1->set_cpu_data(pdfdivcdf_table);

  Blob<Dtype>* blob2 = new Blob<Dtype>(740001, 1, 1, 1);
  Dtype* pdfdivcdf_table_g37 = new Dtype[740001];
  n = 0;
  if (ifile1.is_open()) {
    while (ifile1 >> output) {
      pdfdivcdf_table_g37[n] = output;
      n++;
     }
  }else{
    std::cerr << "There was a problem opening the input file pdfdivcdf_table_g37.txt!\n";
    exit(1);
  }
  blob2->set_cpu_data(pdfdivcdf_table_g37);
  
  //Blob<Dtype>* blob3 = new Blob<Dtype>(2, 1, 1, 1);
  //Dtype* sigma = new Dtype[2];
  //sigma[0] = this->layer_param_.grelu_param().sigma();
  //blob3->set_cpu_data(sigma);
  tables.push_back(blob1);
  tables.push_back(blob2);
  //tables.push_back(blob3);
}

template <typename Dtype>
Dtype GReLULayer<Dtype>::GetRatio(Dtype zeta)
{
  const Dtype* pdfdivcdf_table = tables[0]->cpu_data();
  const Dtype* pdfdivcdf_table_g37 = tables[1]->cpu_data();
  if(zeta < Dtype(-37)){
    if(zeta < Dtype(-4999)){
      return -zeta;
    }else{
      int zeta0 = (int)zeta - 1;
      return -zeta + Dtype(zeta0 + pdfdivcdf_table[-37 - zeta0]);
    }
  }else if(zeta >= Dtype(-37) && zeta <= Dtype(37)){
    return pdfdivcdf_table_g37[(int)(zeta * 10000) + 370000];
  }else{
    return Dtype(0);
  }
}

template <typename Dtype>
void GReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype sigma = this->layer_param_.grelu_param().sigma();//tables[2]->cpu_data()[0];
  for (int i = 0; i < count; ++i) {
	Dtype tmp = bottom_data[i] / sigma;
	if(tmp <= Dtype(-100)){//-100
		top_data[i] = Dtype(-0.000001);
	}else if(tmp > Dtype(37)){
		top_data[i] = bottom_data[i];
	}else{
//    		top_data[i] = std::min(Dtype(100000), bottom_data[i] + sigma * GetRatio(bottom_data[i] / sigma));
	top_data[i] = bottom_data[i] + sigma * GetRatio(bottom_data[i] / sigma);
	}
  }
}

template <typename Dtype>
void GReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const Dtype sigma = this->layer_param_.grelu_param().sigma();//tables[2]->cpu_data()[0];
//	Dtype minv = bottom_data[0] / sigma;
//	Dtype maxv = bottom_data[0] / sigma;
//	Dtype mindf = Dtype(0); Dtype maxdf = Dtype(0);
    for (int i = 0; i < count; ++i) {
	Dtype tmp = bottom_data[i] / sigma;
//	if(tmp > maxv){ maxv = tmp;}
//	if(tmp < minv){minv = tmp;}
	Dtype df = Dtype(0);
	if(tmp <= Dtype(-100)){
		df = Dtype(0);
	}else if(tmp > Dtype(37)){
		df = Dtype(1);
	}else{
      		Dtype ratio = GetRatio(tmp);
		df = Dtype(1) - bottom_data[i] / sigma * ratio - ratio * ratio;
	}
      	bottom_diff[i] = top_diff[i] * df;//(Dtype(1) - bottom_data[i] / sigma * ratio - ratio * ratio);
//	if(i == 0){
//		mindf = df; maxdf = df;
//	}else{
//		if(mindf > df){
//			mindf = df;
//		} 
//		if(maxdf < df){
//			maxdf = df;
//		}
//	}
    }
//	LOG(INFO) << "Maxv = " << maxv << ", Minv = " << minv << ", maxdf = " << maxdf << ", mindf = " << mindf;
  }
}


#ifdef CPU_ONLY
STUB_GPU(GReLULayer);
#endif

INSTANTIATE_CLASS(GReLULayer);
REGISTER_LAYER_CLASS(GReLU);

}  // namespace caffe
