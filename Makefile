TEMPLATE := ~/Work/DATA/SWATI_SPECIES_IN_DIRT/templates/species1.png
LIBRARY := ~/Work/DATA/SWATI_SPECIES_IN_DIRT/db

all : test 
	echo "Test is done"

test : ./detect_species.py $(TEMPLATE) $(LIBRARY)
	rm -rf ./_result/*
	python $< $(TEMPLATE) $(LIBRARY)
