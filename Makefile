
EXECUTABLE=mgrid grid dgxmgrid transfer band

#set use Event to use Event Clock
CODEFLAG=-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 

all: $(EXECUTABLE) 


mgrid: $(CUOBJS) $(COBJS) mgrid_reduction.cu
	nvcc $(CODEFLAG) -std=c++14 -rdc=true -Xcompiler -fopenmp -o $@ $^
grid: $(CUOBJS) $(COBJS) grid_reduction.cu
	nvcc $(CODEFLAG) -rdc=true -std=c++17 -o $@ $^
dgxmgrid:  $(CUOBJS) $(COBJS) mgrid_reduction_dgx1.cu
	nvcc $(CODEFLAG) -std=c++14 -rdc=true -Xcompiler -fopenmp  -o $@ $^
band: bandwidth.cu
	nvcc $(CODEFLAG) -o $@ $^
clean:
	rm  $(EXECUTABLE) 
