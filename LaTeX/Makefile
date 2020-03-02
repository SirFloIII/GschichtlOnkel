MAINTEX = acvrw_template

default: pdf1

all: cleanx pdf clean

pdf:
	pdflatex $(MAINTEX).tex
	bibtex $(MAINTEX)
	pdflatex $(MAINTEX).tex
	pdflatex $(MAINTEX).tex

pdf1:
	pdflatex $(MAINTEX).tex

clean:
	-rm -f $(MAINTEX).aux
	-rm -f $(MAINTEX).bbl
	-rm -f $(MAINTEX).blg
	-rm -f $(MAINTEX).brf
	-rm -f $(MAINTEX).log

cleanx: clean
	-rm -f $(MAINTEX).pdf
