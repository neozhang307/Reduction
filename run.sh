for number in 1 2 4 8 
do
	./grid -b 1 -r 21 -a ${number} -v 2 -w -g>> log/ggrid_w_1.log 
done


for number in 1 2 4 8 
do
	./grid -b 1 -r 21 -a ${number} -v 2 -w -k -g>> log/gkernel_w_1.log 
done


# for number in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
# do
# 	./grid -b 1 -r 21 -a ${number} -v 2 -w >> grid_w_1.log 
# done


# for number in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
# do
# 	./grid -b 1 -r 21 -a ${number} -v 2 -w -k >> kernel_w_1.log 
# done

for number in 2 4 8 20 200 2000
do
        ./grid -b 1 -r 21 -a ${number} -v 2 >> log/grid_bigkernel.log
        ./grid -b 1 -r 21 -a ${number} -v 2 -k >> log/grid_kernels.log
done

for number in 2 4 10 100 1000
do
        ./grid -b 2 -r 21 -a ${number} -v 2 -k >> log/grid_kernels_over.log
done