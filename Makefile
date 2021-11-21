all:
	./regression.sh rebuild
	cp build/winograd .

clean:
	./regression.sh clean
