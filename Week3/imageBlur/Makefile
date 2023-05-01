CC=nvcc 

ch03.out:  mkClockMeasure.h  mkCuda.h  mkPpm.h main.cu
	$(CC) -o ch03.out main.cu

clean:
	rm -rf ch03.out *.ppm
