all:
	naluX -i input.yaml

clean:
	rm -f slurm* *log

purge: clean
	rm -rf res*  *.csv

purgeaoa:
	rm -rf *aoa*

aoa:
	nalu-aseq input_turb.yaml -a -10 20 1 -b submit-unity.sh  -j dutu --submit
	nalu-aseq input_trans.yaml -a -10 20 1 -b submit-unity.sh -j dutr --submit

submit:
	for file in submit-input_*.sh ; do  sbatch $$file; done
