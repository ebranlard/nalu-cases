all:
	naluX -i input.yaml

clean:
	rm -f slurm* *.log

purge: clean
	rm -rf res*  *.csv submit*_aoa*.sh input*_aoa*.yaml

aoa:
	nalu-aseq input.yaml -a -40 40 5 -b submit-unity.sh -j pov -v

submit:
	for file in submit-input_*.sh ; do  sbatch $$file; done
