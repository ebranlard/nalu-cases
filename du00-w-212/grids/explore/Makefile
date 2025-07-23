all: match

prep:
	exo-layers.exe ../du00w212_re3M_y_aoa0_n1_original.exo --airfoil -n 500 > ./LayersInfo.txt --no-check
	cat  ./LayersInfo.txt | grep "Layer    0:"
	cat  ./LayersInfo.txt | grep "Layer  143:"
	mv ../du00w212_re3M_y_aoa0_n1_original_airfoil.txt ./du00-w-212.txt
	arf-convert.exe -i ./du00-w-212.txt -o ./du00-w-212.fmt --thick

match:
	pyhyp.exe --s0 2.8e-7 --marchDist 105 -n 145 -i ../../../airfoils/du00-w-212.csv -o ../du00w212_re3M_y03_aoa0_n1.exo
#	exo-layers.exe ../nlf1-0416_re4M_y2_aoa0_n1_NEW.exo --airfoil -n 500 > ./nlf0416_F_Manu_layersINFO.txt

