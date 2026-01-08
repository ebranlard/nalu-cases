all:
	naluX -i input.yaml

clean:
	rm -f slurm* *log

purge: clean
	rm -rf res*  *.csv

purgeaoa:
	rm -rf *aoa*

aoa:
	nalu-aseq input.yaml -a -10 20 1 -b submit-unity.sh -j dupo --no-check

submit:
	for file in submit-input_*.sh ; do  sbatch $$file; done
