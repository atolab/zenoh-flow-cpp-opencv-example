.PHONY : example fresh clean all

all: example

clean:
	rm -rf op *.so

fresh: clean all

example:
	mkdir op
	cd op && cmake .. -DOPERATOR=ON  && make && mv libcxx_operator.so ../libopencv_dectect_op.so && make clean && cd ..