.PHONY: module_load
module_load:
	module load vim cmake gcc cuda/12.1.0 python/3.11.2	

10_cavity.out: 10_cavity.cu
	nvcc -DNDEBUG 10_cavity.cu -o 10_cavity.out	

.PHONY: run_10_cavity_cu
run_10_cavity_cu: 10_cavity.out
	./10_cavity.out | python3 10_cavity_plot.py

