all:
	cc main.c rply.c glad_gl.c -I inc -Ofast -lglfw -lm -o plv

win:
	i686-w64-mingw32-gcc main.c rply.c glad_gl.c -I inc -Ofast -L. -lglfw3dll -lm -o PLYViewer.exe

clean:
	rm -f plv
	rm -f PLYViewer.exe