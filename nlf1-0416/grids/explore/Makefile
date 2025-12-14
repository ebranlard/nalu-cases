all: match

prep:
	exo-layers.exe ../nlf1-0416_re4M_y2_aoa0_n1-Original.exo --airfoil -n 500 > ./nlf0416_F_layersINFO.txt
	cat  ./nlf0416_F_layersINFO.txt | grep "Layer    0:"
	mv ../nlf1-0416_re4M_y2_aoa0_n1-Original_airfoil.txt ./nlf0416_F_airfoil.txt
	arf-convert.exe -i ./nlf0416_F_airfoil.txt -o ./nlf0416_F_airfoil.fmt --thick

match:
	pyhyp.exe --s0 2.1e-6 --marchDist 1100 -n 145 -i .\nlf0416_F_airfoil.fmt -o ../nlf1-0416_re4M_y2_aoa0_n1_NEW.exo
	exo-layers.exe ../nlf1-0416_re4M_y2_aoa0_n1_NEW.exo --airfoil -n 500 > ./nlf0416_F_Manu_layersINFO.txt

re:
	pyhyp.exe --re 4e6 --marchDist 1100 -n 145 -i .\nlf0416_F_airfoil.fmt -o .\nlf0416_F_Manu_Re.exo
