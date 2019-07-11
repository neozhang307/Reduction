
CUSOURCE= block_reduction.cu warp_reduction.cu bandwidth.cu
CUOBJS=$(CUSOURCE:.cu=.o)

EXECUTABLE=warp block band 

all: $(EXECUTABLE) 

band: bandwidth.cu
	nvcc -arch sm_60 -O3 -o $@ $^

warp: warp_reduction.cu
	nvcc -arch sm_60 -O3 -o $@ $^

block: block_reduction.cu
	nvcc -arch sm_60 -O3 -o $@ $^

clean:
		rm $(EXECUTABLE)