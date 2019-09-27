
EXECUTABLE=mgrid dgxmgrid transfer band

#set use Event to use Event Clock
CODEFLAG=-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 

all: $(EXECUTABLE) 


mgrid: $(CUOBJS) $(COBJS) mgrid_reduction.cu
	nvcc $(CODEFLAG) -std=c++11 -rdc=true -Xcompiler -fopenmp -o $@ $^
dgxmgrid:  $(CUOBJS) $(COBJS) mgrid_reduction_dgx1.cu
	nvcc $(CODEFLAG) -std=c++11 -rdc=true -Xcompiler -fopenmp  -o $@ $^
transfer: $(CUOBJS) $(COBJS) transfer.cu
	nvcc $(CODEFLAG) -std=c++11 -rdc=true -o $@ $^
band: bandwidth.cu
	nvcc -arch sm_60 bandwidth.cu -o $@ $^
clean:
	rm  $(EXECUTABLE) 
