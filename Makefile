MAKEFILE=
ifeq (x$(TARGET),xCUDA)
	MAKEFILE = Makefile_cuda
endif
ifeq (x$(TARGET),xHIP)
	MAKEFILE = Makefile_hip
endif
ifeq (x$(MAKEFILE),x)
	MAKEFILE = Makefile_cuda
endif

all:
	make -f $(MAKEFILE)

clean:
	make -f $(MAKEFILE) clean

