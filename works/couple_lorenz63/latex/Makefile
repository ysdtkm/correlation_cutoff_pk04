topt = --halt-on-error --no-pdf
src  = main.tex
obj  = out.pdf

all: $(obj)

$(obj): $(src)
	xelatex $(topt) $(src)
	xelatex $(topt) $(src)
	echo $(src) | sed -e 's/.tex/.xdv/g' | xargs -i xdvipdfmx -o $(obj) {}
	echo $(src) | sed -e 's/.tex/.xdv/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.aux/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.log/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.ps/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.blg/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.bbl/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.out/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.snm/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.toc/g' | xargs rm -f
	echo $(src) | sed -e 's/.tex/.nav/g' | xargs rm -f

clean:
	rm -rf $(obj)

.PHONY: all clean

