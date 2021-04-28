#snakemake -j --use-conda # local execution
#snakemake --profile axon -j --use-conda --use-envmodules # slurm exectution on axon

configfile: "config.yml"
root = config['local_data_path']

from pathlib import Path

localrules:
    merge_predicted_probabilities,
    produce_metrics,
    plot_metrics


rule all:
    input:
        expand(root + '/data_out/tutorial_ECS/xfold/M0007_plots/1000.fig',
            ident=['M0007', 'M0027']),


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
       root + '/data_out/tutorial_ECS/xfold/{ident}_output.mat'
    input:
        h5_raw_data_path = '/axon/scratch/pwatkins/datasets/raw/{ident}_33_39x35x7chunks_Forder.h5',
        lblsh5 = root + '/data_out/tutorial_ECS/xfold/{ident}_supervoxels.h5',
        skelin = '/soma/soma_fs/cne/pwatkins/cne_nas_bkp/from_externals/ECS_paper/skeletons/{ident}_33_dense_skels.152.nml',
    params:
        chunk = lambda wc: config['datasets'][wc.ident]['chunk'],
    envmodules:
        'matlab/R2020b'
    shell:
        """matlab -nojvm -nosplash -batch "addpath(genpath('recon/matlab')); knossos_efpl_top_snakemake('{output}', '{input.lblsh5}', '{input.h5_raw_data_path}', '{input.skelin}', [{params.chunk}])" """

rule plot_metrics:
    output:
        fig1000 = root + '/data_out/tutorial_ECS/xfold/{ident}_plots/1000.fig',
        fig1001 = root + '/data_out/tutorial_ECS/xfold/{ident}_plots/1001.fig',
        fig1002 = root + '/data_out/tutorial_ECS/xfold/{ident}_plots/1002.fig',
        fig1003 = root + '/data_out/tutorial_ECS/xfold/{ident}_plots/1003.fig',
    input:
        input_mat = root + '/data_out/tutorial_ECS/xfold/{ident}_output.mat'
    params:
        output_path = lambda wc: root + f'/data_out/tutorial_ECS/xfold/{wc.ident}_plots',
    envmodules:
        'matlab/R2020b'
    shell:
        """matlab -nosplash -batch "addpath(genpath('recon/matlab')); knossos_efpl_plot_top_snakemake('{params.output_path}', '{input.input_mat}')" """