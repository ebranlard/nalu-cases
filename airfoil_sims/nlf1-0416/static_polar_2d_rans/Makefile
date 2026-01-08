all:
	naluX -i input.yaml

clean:
	rm -f slurm* *log

purge: clean
	rm -rf res*  *.csv

purgeaoa:
	rm -rf *aoa*

aoa:
	nalu-aseq input.yaml -a -7 20 1 -b submit-unity.sh -j Fpo

submit:
	for file in submit-input_*.sh ; do  sbatch $$file; done
