

mesh2:
	pyhyp --re 2e6 --marchDist 75 -n 145 -i ../../airfoils/S809_l600.csv -o ./S809_l600_m145_n1_re2M_y1.2mu.exo --yplus 0.1 --verbose

mesh10:
	pyhyp --re 1e7 --marchDist 75 -n 145 -i ../../airfoils/S809_l600.csv -o ./S809_l600_m145_n1_re10M_y0.27mu.exo --yplus 0.1 --verbose

mesh1:
	pyhyp --re 1e6 --marchDist 75 -n 145 -i ../../airfoils/S809_l600.csv -o ./S809_l600_m145_n1_re1M_y2.3mu.exo --yplus 0.1 --verbose

mesh01:
	pyhyp --re 0.1e6 --marchDist 75 -n 145 -i ../../airfoils/S809_l600.csv -o ./S809_l600_m145_n1_re0.1M_y19mu.exo --yplus 0.1 --verbose





#
extractold:
	exo-layers --airfoil -n 2 ./pointwise/S809_Re=0.1M_AoA0_nSpan=1.exo

