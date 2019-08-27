
EXECUTABLE=mgrid transfer

#set use Event to use Event Clock
CODEFLAG=-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 

all: $(EXECUTABLE) 


mgrid: $(CUOBJS) $(COBJS) mgrid_reduction.cu
	nvcc $(CODEFLAG) -std=c++11 -rdc=true -o $@ $^

transfer: $(CUOBJS) $(COBJS) transfer.cu
	nvcc $(CODEFLAG) -std=c++11 -rdc=true -o $@ $^

clean:
	rm  $(EXECUTABLE) 
