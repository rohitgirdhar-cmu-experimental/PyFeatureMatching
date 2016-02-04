# Set the following:
CAFFE_ROOT := /home/rgirdhar/Software/vision/caffe_new/caffe_cpu
CUDA_ROOT := /usr/local/cuda
#INC += -I$(CAFFE_INSTALL_DIR)/gflags_install/include -I$(CAFFE_INSTALL_DIR)/glog_install/include -I$(CAFFE_INSTALL_DIR)/mdb_install/include -I ~/software/boost/include
#LIBS += -L$(CAFFE_INSTALL_DIR)/gflags_install/lib -L$(CAFFE_INSTALL_DIR)/glog_install/lib -L$(CAFFE_INSTALL_DIR)/mdb_install/lib -L ~/software/boost/libs
CPU_ONLY := 1

CXXFLAGS += -std=c++0x -fopenmp -O2 -g
LIBS += -L$(CAFFE_ROOT)/build/lib -L/home/ashrivas/cv-libs/cuda-5.5/lib64/ -L/home/rgirdhar/Software/utils/boost/lib/ -L/home/rgirdhar/Software/utils/hdf5/hdf5-1.8.13-linux-x86_64-shared/lib/
LDFLAGS += -lcaffe -lglog -lboost_program_options -lboost_filesystem -lboost_system -lboost_serialization -llmdb `pkg-config --libs /home/ashrivas/cv-libs/opencv/lib/pkgconfig/opencv.pc` -lhdf5 -lzmq
INC += -I$(CAFFE_ROOT)/include/ -I$(CAFFE_ROOT)/build/src/ -I/home/rgirdhar/Software/utils/boost/include/ -I/home/rgirdhar/Software/utils/hdf5/hdf5-1.8.13-linux-x86_64-shared/include/
ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CXXFLAGS += -DCPU_ONLY
endif
CXXFLAGS += -DUSE_OPENCV

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile utils.hpp Config.hpp
	$(CXX)  \
		$(CXXFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) 

clean:
	rm -rf *.bin
