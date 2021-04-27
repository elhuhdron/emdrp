#snakemake -j --use-conda # local execution
#snakemake --profile axon -j --use-conda # slurm exectution on axon

configfile: "config.yml"
root = config['local_data_path']

from pathlib import Path

rule all:
    input:
        #expand( root + '/data_out/tutorial_ECS/xfold/{ident}_supervoxels.h5', 
        #    ident=['M0007', 'M0027']),
        #    ident=['M0007']),
        'matlab_script.chkpt',

rule merge_predicted_probabilities:
    output:
        root + '/data_out/tutorial_ECS/xfold/{ident}_probs.h5',
    input:
        expand(root + '/data_out/tutorial_ECS/xfold/{{ident}}_{replicate}.0_probs.h5',
            replicate = glob_wildcards(root + '/data_out/tutorial_ECS/xfold/{{ident}}_{replicate}.0_probs.h5').replicate),
    params:
        src_path = root + '/data_out/tutorial_ECS/xfold/',
        srcfiles = lambda wc: [p.name for p in Path(root + '/data_out/tutorial_ECS/xfold/').glob(f'{wc.ident}_*.0_probs.h5')],
        dim_order = lambda wc: ['xyz' for p in Path(root + '/data_out/tutorial_ECS/xfold/').glob(f'{wc.ident}_*.0_probs.h5')],
        size = lambda wc: config['datasets'][wc.ident]['size'],
        chunk = lambda wc: config['datasets'][wc.ident]['chunk'],
    resources:
        time='00:05:00', 
        partition="p.default",
        #gres="gpu:rtx2080ti:1",
        mem="32000",
    conda:
        'environment.yml'
    shell:
        'python -u recon/emdrp/dpMergeProbs.py' +
        ' --srcpath {params.src_path}' +
        ' --srcfiles {params.srcfiles}' +
        ' --dim-orderings {params.dim_order}' +
        ' --outprobs {output}' +
        ' --chunk {params.chunk}' +
        ' --size {params.size}' +
        ' --types ICS ECS MEM --ops mean min --dpMergeProbs-verbose'

rule apply_watershed_on_ICS_probability:
    output:
         root + '/data_out/tutorial_ECS/xfold/{ident}_supervoxels.h5',
    input:
        root + '/data_out/tutorial_ECS/xfold/{ident}_probs.h5'
    params:
        size = lambda wc: config['datasets'][wc.ident]['size'],
        chunk = lambda wc: config['datasets'][wc.ident]['chunk'],
    resources:
        time='12:00:00',
        partition="p.axon",
        mem="32000",
        cpus_per_task="2",
    conda:
        'environment.yml'
    shell:
        'python -u recon/emdrp/dpWatershedTypes.py' +
        ' --probfile {input}' +
        ' --chunk {params.chunk} --offset 0 0 0 --size {params.size}' +
        ' --outlabels {output}' +
        ' --ThrRng 0.5 0.999 0.1' +
        ' --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999 --dpW'

rule produce_metrics:
    output:
        touch('matlab_script.chkpt')
    envmodules:
        'matlab/R2019b'
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('recon/matlab')); knossos_efpl_top_snakemake()" """

