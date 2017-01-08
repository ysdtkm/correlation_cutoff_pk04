# src: all dependencies (at least one from each group of files)
# obj: one of surely-updated files (no wildcard)
# cln: all files which must be cleaned (wildcard allowed)
# cmd: any execution command

src1 = Py/main.py Py/const.py
obj1 = data/xyz_short_fcst.bin
cln1 = data/*.bin
cmd1 = python Py/main.py

src2 = Py/plot.py Py/const.py $(obj1)
obj2 = image/xyz_short_1.png
cln2 = image/*
cmd2 = python Py/plot.py

all: $(obj1) $(obj2)

$(obj1): $(src1)
	$(cmd1)

$(obj2): $(src2)
	$(cmd2)

clean:
	rm -rf $(cln1) $(cln2)

.PHONY: all clean

