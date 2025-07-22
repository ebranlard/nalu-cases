


all: meshes


meshes:
	pyhyp --s0 2.8e-7 --marchDist 105 -n 145 -i ../../airfoils/du00-w-212.csv -o ./du00w212_re3M_y03_aoa0_n1.exo --verbose
	# NOTE: Re needs y+ 0.035 to match s0=2.8e-7
	pyhyp --re 3e6 --marchDist 105 -n 145 -i ../../airfoils/du00-w-212.csv -o ./du00w212_re3M_y03_aoa0_n1_re.exo --yplus 0.1 --verbose

meshre:
	pyhyp --re 3e6 --marchDist 105 -n 145 -i ../../airfoils/du00-w-212.csv -o ./du00w212_re3M_y03_aoa0_n1_re.exo --yplus 0.035 --verbose

