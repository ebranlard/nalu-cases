all:
	naluX -i input.yaml

clean:
	rm -f slurm* *.log
	rm -rf *aoa*.sh
	rm -rf *aoa*.yaml
	rm -rf meshes

purge: clean
	rm -rf res*  
	rm -rf *.csv

aoa:
	nalu-aseq input_turb.yaml  -a -7 20 1 -b submit-unity.sh -j stu1 --submit
	nalu-aseq input_trans.yaml -a -7 20 1 -b submit-unity.sh -j str1 --submit

submit:
	for file in submit-input_*.sh ; do  sbatch $$file; done
